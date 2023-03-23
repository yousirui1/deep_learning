def log_mel_spectrogram(audio_file, param):
    wav_data, sr = sf.read(audio_file, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype

    # 补0 to do  mask
    if len(wav_data) < param.sample_rate:
        wav_data = np.pad(wav_data, (0, int(param.sample_rate - len(wav_data))), 'constant', constant_values = 0)

    waveform = wav_data / 32768.0

    if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)  # 多通道转单通道

    if sr != param.sample_rate:
            waveform = resampy.resample(waveform, sr, param.sample_rate)

    waveform = np.reshape(waveform, [1, -1]).astype(np.float32)
    return features_lib.waveform_to_log_mel_spectrogram_patches(tf.squeeze(waveform, axis=0), param)


class MineDataSet():
    def __init__(self, params, path, cache_dir = None, file_type='wav', train_split = 0.9, wanted_label = None):
        self.params = params
        self.path = path
        if cache_dir is None:
            self.cache_dir = path
        else:
            self.cache_dir = cache_dir
        self.file_type = file_type
        self.train_split = train_split
        self.wanted_label = wanted_label

    def __save_labels(self, path, labels):
        with open(path + 'mine_classes.json', 'w') as f:
            json.dump(labels, f)
        print('Saved model architecture')

    def __build_cache(self, file_list, labels, cache_path, params):
        total_size = len(file_list)
        print(len(labels))
        y = np.zeros((total_size, len(labels)), dtype=np.int64)   # batch_size
        with tf.io.TFRecordWriter(cache_path) as writer:
            for index, audio_file in enumerate(file_list):
                class_id = audio_file.split('/')[-2]
                spectrogram, patches = log_mel_spectrogram(audio_file, params)
                y[index, [labels[class_id]]] = 1 
                example = audio_example(patches.numpy(), y[index])
                writer.write(example.SerializeToString())
                d = f"Scanning '{audio_file}' audio and labels... {total_size} found, {index} corrupted"
                tqdm(None, desc=d, total=total_size, initial=index)  # display cache results
            writer.close()
        #print('Saved model architecture')
            
    def build_dataset(self):
        files_train, files_valid, labels = get_files_and_labels(self.path + 'train_wav/', self.file_type, self.train_split, self.wanted_label)
        self.__build_cache(files_train, labels, self.cache_dir + 'train.cache', self.params)
        self.__build_cache(files_valid, labels, self.cache_dir + 'valid.cache', self.params)
        self.__save_labels(self.path, labels)



def _parse_audio_function(example_string):
    n_classes = 527 
    feature = { 
        'patches': tf.io.FixedLenFeature([], tf.string),
        'patches_shape': tf.io.FixedLenFeature(shape=(3,), dtype=tf.int64), # shape = 3 
        'label': tf.io.FixedLenFeature([n_classes], dtype=tf.int64), 
    }    
    feature_dict = tf.io.parse_single_example(example_string, feature)
    patches_raw = feature_dict['patches']   
    patches_shape = feature_dict['patches_shape']
    label = feature_dict['label']

    patches = tf.io.decode_raw(patches_raw, tf.float32)
    patches = tf.reshape(patches, patches_shape)
    label = tf.reshape(label, (1, n_classes)) 
    return patches, label


def audio_example(patches, label): # tf record example
    feature = { 
        'patches': tools._numpy_float32_feature(patches),
        'patches_shape': tools._shape_feature(patches.shape),
        'label': tools._int64_list_feature(label),
    }   
    return tf.train.Example(features=tf.train.Features(feature=feature)) 

class AudioSetDataSet():
    def __init__(self, params, path, cache_dir = None):
        self.params = params
        self.path = path
        self.cache_dir = cache_dir

    def __read_pd(self, path):
        valid_meta = pd.read_csv(path + 'valid.csv')
        valid_id = valid_meta.groupby('YTID')['YTID'].apply(lambda cat: cat.sample(1)).reset_index()['YTID']
        train_meta = pd.read_csv(path+ 'train.csv')
        train_id = train_meta.groupby('YTID')['YTID'].apply(lambda cat: cat.sample(1)).reset_index()['YTID']
        class_meta = pd.read_csv(path + 'class_labels_indices.csv')
        class_id = class_meta.groupby('index')['index'].apply(lambda cat: cat.sample(1)).reset_index()['index']
        return valid_meta, valid_id, train_meta, train_id, class_meta, class_id

    def __build_classes(self, class_meta, class_id):
        labels = []
        label_index = {}
        for index in range(len(class_id)):
            item = []
            clas = class_id[index]
            clas = class_meta[class_meta.index == clas]
            label_index[clas.mid.to_string(index=False)] = clas.index[0]
            item.append(clas.index[0])
            labels.append(item)

        mlb = MultiLabelBinarizer()
        mlb.fit(labels)
        print(labels)
        return label_index, mlb, mlb.classes_

    def __build_cache(self, wav_dir, cache_path, file_ytid, meta,  mlb, label_index, params):
        total_size = len(file_ytid)
        with tf.io.TFRecordWriter(cache_path) as writer:
            for index in range(total_size):
                label_id = []
                item = file_ytid[index]
                item = meta[meta.YTID == item]

                multi_label = item.positive_labels.to_string(index=False).split(",")
                for label in multi_label:
                    label_id.append(label_index[label])

                wav_path = wav_dir + item.YTID.to_string(index=False) + '.wav'
                spectrogram, patches = log_mel_spectrogram(wav_path, params)
                example = audio_example(patches.numpy(), mlb.transform([label_id])[0])
                writer.write(example.SerializeToString())
                d = f"Scanning '{wav_path}' audio and labels... {total_size} found, {index} corrupted"
                tqdm(None, desc=d, total=total_size, initial=index)  # display cache results

            writer.close()

    def build_dataset(self):
        valid_meta, valid_id, train_meta, train_id, class_meta, class_id = self.__read_pd(self.path)
        label_index, mlb, classes = self.__build_classes(class_meta, class_id)
        self.__build_cache(self.path + 'train_wav/', self.path + 'train.cache', train_id, train_meta, mlb,
                                    label_index, self.params)
        self.__build_cache(self.path + 'valid_wav/', self.path + 'valid.cache', valid_id, valid_meta, mlb,
                                    label_index, self.params)

class MineDataSet():
    def __init__(self, params, path, cache_dir = None, file_type='wav', train_split = 0.9, wanted_label = None):
        self.params = params
        self.path = path
        if cache_dir is None:
            self.cache_dir = path
        else:
            self.cache_dir = cache_dir
        self.file_type = file_type
        self.train_split = train_split
        self.wanted_label = wanted_label

    def __save_labels(self, path, labels):
        with open(path + 'mine_classes.json', 'w') as f:
            json.dump(labels, f)
        print('Saved model architecture')

    def __build_cache(self, file_list, labels, cache_path, params):
        total_size = len(file_list)
        print(len(labels))
        y = np.zeros((total_size, len(labels)), dtype=np.int64)   # batch_size
        with tf.io.TFRecordWriter(cache_path) as writer:
            for index, audio_file in enumerate(file_list):
                class_id = audio_file.split('/')[-2]
                spectrogram, patches = log_mel_spectrogram(audio_file, params)
                y[index, [labels[class_id]]] = 1
                example = audio_example(patches.numpy(), y[index])
                writer.write(example.SerializeToString())
                d = f"Scanning '{audio_file}' audio and labels... {total_size} found, {index} corrupted"
                tqdm(None, desc=d, total=total_size, initial=index)  # display cache results
            writer.close()
        #print('Saved model architecture')

    def build_dataset(self):
        files_train, files_valid, labels = get_files_and_labels(self.path + 'train_wav/', self.file_type, self.train_split, self.wanted_label)
        self.__build_cache(files_train, labels, self.cache_dir + 'train.cache', self.params)
        self.__build_cache(files_valid, labels, self.cache_dir + 'valid.cache', self.params)
        self.__save_labels(self.path, labels)




    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 2
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    search_path = os.path.join(self.data_dir, '*', '*/*.wav')
    for wav_path in gfile.Glob(search_path):
      word,_ = os.path.split(os.path.dirname(wav_path))
      word = word.lower()
      word_tmp = word
      word_tmp = word_tmp.split('/')
      word = word_tmp[1]
      # Treat the '_background_noise_' folder as a special case, since we expect
      # it to contain long audio samples we mix in to improve training.
      if word == BACKGROUND_NOISE_DIR_NAME:
        continue
      all_words[word] = True
      set_index = which_set(wav_path, validation_percentage, testing_percentage)
      # If it's a known class, store its detail, otherwise add it to the list
      # we'll use to train the unknown label.
      if word in wanted_words_index:
        self.data_index[set_index].append({'label': word, 'file': wav_path})
        #print("1111"+wav_path)
      else:
        unknown_index[set_index].append({'label': word, 'file': wav_path})
        #print("222"+wav_path)
    if not all_words:
      raise Exception('No .wavs found at ' + search_path)
    for index, wanted_word in enumerate(wanted_words):
      if wanted_word not in all_words:
        raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    print(self.data_index)
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
      # Pick some unknowns to add to each partition of the data set.
      random.shuffle(unknown_index[set_index])
      unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
      self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list(wanted_words)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX


class LoadAudioSet():
    def __init__(self, path, params, single_cls=False, rect=False, prefix=''):
        self.path = path
        self.params = params
        self.rect = rect

        #try:
        #    f = [] #audio files
        #    for p in path if isinstance(path, list) else [path]:
        #        p = Path(p) # os-agnostic
        #        if p.is_dir():
        #            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
        #        elif p.is_file(): # file  .txt
        #            if sorted([p.replace('/', os.sep) for p in f if x.split('.')[-1].lower() in audio_formats]):
        #                with open(p, 'r') as t:
        #                    t = t.read().strip().splitlines()
        #                    parent = str(p.parent) + os.sep
        #                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
        #        else:
        #            raise Exception(f'{prefix}{p} does not exist')

        #    self.audio_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in audio_formats])
        #    print(len(self.audio_files))
        #    assert self.audio_files, f'{prefix} No audio found'
        #except Exception as e:
        #    raise Exception(f'{prefix}Error loading data from {path}: {e}\n')

        # Check cache 
        self.label_files = 'valid'
        cache_path = (p if p.is_dir() else Path(self.label_files)).with_suffix('.cache') # to do

        if cache_path.is_file():
            cache, exists = self.load_cache(str(cache_path)), True
        else:
            cache, exists  = self.cache_data(str(cache_path)), False

        # Displya cache
        #cache('result') # found, missing, empty, corrupted, total
        #print()
        if exists:
            print('')
            #d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            #tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results

        if single_cls:
            print('single_cls') # 单标签  to do 
            #for x in 

        # Recangular Training
        if self.rect:
            print('rect !!')  #to do
    def audio_example(self, patches, label): # tf record example
        feature = {
            'patches': _numpy_float32_feature(patches),
            'patches_shape': _shape_feature(patches.shape),
            'label': _numpy_int32_feature(label),
            'label_shape': _shape_feature(label.shape),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _parse_audio_function(self,example_proto):
        feature = {
            'patches': tf.io.FixedLenFeature([], tf.string),
            'patches_shape': tf.io.FixedLenFeature(shape=(3,), dtype=tf.int64), # shape = 3 
            'label': tf.io.FixedLenFeature([], tf.string),
            'label_shape': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64), #shape (1,)
        }
        return tf.io.parse_single_example(example_proto, feature)

    def load_cache(self, cache_path):
        raw_audio_dataset = tf.data.TFRecordDataset(cache_path)
        audio_cache = raw_audio_dataset.map(self._parse_audio_function)
        return audio_cache

    def cache_data(self, cache_path):
        param = yamnet_params.Params()
        count = 0
        with tf.io.TFRecordWriter(cache_path) as writer:
            for audio_file in self.audio_files:
                #d = f"total len {len(self.audio_files)} len {count}"
                print(f"total len {len(self.audio_files)} len {count}")
                #print(d)
                count += 1
                spectrogram, patches = log_mel_spectrogram(audio_file, param)
                example = self.audio_example(patches.numpy(), 0)
                writer.write(example.SerializeToString())
        return self.load_cache(cache_path)

    def __len__(self):
        return self.cache.size()

