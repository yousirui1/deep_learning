import tensorflow as tf
import os 
os.sys.path.append('./audioset/yamnet')
import params as yamnet_params
import features as features_lib
import soundfile as sf
import numpy as np
import resampy
from pathlib import Path
import glob
from tqdm import tqdm
from tensorflow.python.keras.utils.data_utils import Sequence
from random import shuffle

audio_formats = ['wav', 'aac', 'pcm']
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
SILENCE_LABEL = '_Silence_'
UNKNOWN_WORD_LABEL = '_Unknown_'

# PCM
class LoadSounds:
    print("")
    #def __initfor k in range(ts_len):
    #ts_file = ts_list[k]
    #file_name = 



# RGB
class LoadImages:
    print("")
    #def __init__(self, path, img_size = 640, stride=32):
        #p = str(Path(path).absolute())      # os-agnostic absolute path
        #if '*' in p:
            #ZZ


def _bytes_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _numpy_int32_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.reshape(-1)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(np.float32).tostring()]))


def _numpy_float32_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.reshape(-1)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(np.float32).tostring()]))

def _numpy_float64_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.reshape(-1)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(np.float64).tostring()]))

def _float_feature(value):
    # Returns a float_list from a float / double 
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    # Returns a int64_list from a bool / enum / int /uint 
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _shape_feature(value):
    # Returns a int64_list from a bool / enum / int /uint 
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def log_mel_spectrogram(audio_file, param):
    wav_data, sr = sf.read(audio_file, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype

    if len(wav_data) < param.sample_rate:
        # 补0 to do  mask
        wav_data = np.pad(wav_data, (0, param.sample_rate - len(wav_data)), 'constant', constant_values = 0)

    waveform = wav_data / 32768.0

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)  # 2 维转 3维 

    if sr != param.sample_rate:
        waveform = resampy.resample(waveform, sr, param.sample_rate)

    waveform = np.reshape(waveform, [1, -1]).astype(np.float32)
    return features_lib.waveform_to_log_mel_spectrogram_patches(tf.squeeze(waveform, axis=0), param)


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

def get_files_and_labels(train_dir, typ='wav', train_split=0.9, wanted_label = None):
    #ignored = {"folder_one", "folder_two", "folder_three"}
    #folders = [x for x in os.listdir(path) if x not in ignored]
    
    if not wanted_label:
        classes = sorted(os.listdir(train_dir))
    else :
        classes = [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_label.split(',') 
    files_train = list()
    files_val = list()
    labels = dict()
 
    for cnt, i in enumerate(classes): # loop over classes
        tmp = os.listdir(train_dir+i)
        shuffle(tmp)
        for j in tmp[:round(len(tmp)*train_split)]: # loop over training samples
            if j.split('.')[-1]==typ:
                files_train.append(train_dir + i +'/' + j)
        for j in tmp[round(len(tmp)*train_split):]: # loop over validation samples
            if j.split('.')[-1]==typ:
                files_val.append(train_dir + i +'/' + j)
        labels[i] = cnt
    return files_train, files_val, labels
     
class DataGenerator_NP(Sequence):
    
    'Generates YAMNet patches'
    def __init__(self, 
                 list_IDs, 
                 labels, 
                 dim = (96, 64),
                 batch_size=1, 
                 n_classes=3,
                 use_background = 0,
                 data_dir = '',
                 shuffle=True,
                 class_weights=None):
        'Initialization'
        self.classes = []
        self.batch_size = batch_size
        self.labels = labels
        self.dim = dim
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.class_weights = class_weights
        self.use_background = use_background
        self.data_dir = data_dir
        self.prepare_background_data()
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        if self.class_weights:
            X, y, sample_weights = self.__data_generation(list_IDs_temp)
            return X, y, sample_weights
        else:    
            X, y = self.__data_generation(list_IDs_temp)
            return X, y

    def get_data(self):
        x = 1
        y = 1
        return x, y
    
    def prepare_background_data(self):
        self.background_data = []
        
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        #if not gfile.Exists(background_dir):
        if os.path.exists(background_dir) == False:
            print("no background audio")
            self.use_background = False
            return self.background_data
        search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME, '*.wav')
        
        #desired_samples = 16000
        #sample_rate = 16000
        for wav_file in sorted(glob.glob(os.path.join(search_path))):
            print(wav_file)
            
            wav_data, sr = sf.read(wav_file, dtype=np.int16)
            assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
       
            '''
            if len(wav_data) < 16000:
                wav_data = np.pad(wav_data, (0, 16000 - len(wav_data)), 'constant', constant_values = 0)
            print(len(wav_data))
            waveform = wav_data / 32768.0
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
            
            if sr != params.sample_rate:
                waveform = resampy.resample(waveform, sr, params.sample_rate)
            
            waveform = np.reshape(waveform, [1, -1]).astype(np.float32)
            spectrogram, patches = features_lib. \
                          waveform_to_log_mel_spectrogram_patches(tf.squeeze(waveform, axis=0), 
                                                      params)
            
            wav_data = ess.MonoLoader(filename = wav_file, sampleRate = sample_rate)() 
            wav_pos = 0
            while True:     
                if len(wav_data) >=  desired_samples:
                    audio = wav_data[:desired_samples]
                    wav_pos = wav_pos + desired_samples
                    wav_data = wav_data[wav_pos:]
                    self.background_data.append(audio)
                    if len(wav_data) == 0:
                        break
                elif len(wav_data) >= desired_samples // 2:
                    audio = np.pad(wav_data, (0, desired_samples - len(wav_data)), 'constant', constant_values = 0)
                    self.background_data.append(audio)
                    break
                else:
                    break 
            '''
        if not self.background_data:
            raise Exception('No background wav files were found in ' + search_path)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization   

        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes))
        sample_weights = np.empty((self.batch_size, ))
        
        y[:] = 0
        #print(list_IDs_temp)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #print(ID)            
            class_id = ID.split('/')[-2]
            #print(class_id)
            y[i,self.labels[class_id]] = 1
            #print(y)          
            sample = np.load(ID)
            
            # if the waveform for this sample was long enough to contain multiple patches, randomly select one of the patches
            if sample.shape[0] > 1:
                sample = np.squeeze(sample[np.random.choice(range(sample.shape[0]), 1)])
            '''
            if self.use_background and number_of_certain_probability():
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                #print(background_samples)
                if len(background_samples) < self.model_settings['desired_samples']:
                      raise ValueError(
                          'Background sample is too short! Need more than %d'
                      ' samples but only %d were found' %
                      (self.model_settings['desired_samples'], len(background_samples)))
                sample = sample + (background_volume * background_reshaped).flatten()
            '''    
            X[i,] = sample
                
            if self.class_weights:
                sample_weights[i] = self.class_weights[self.labels[class_id]]
          
        self.classes.append(y.reshape(y.shape[1]))

        if self.class_weights is not None:
            return X, y, sample_weights
        else:
            return X, y

class DataGenerator(Sequence):
    def __init__(self, cache_dir, batch_size = 1, train_split = 1, dim = (96, 64), n_classes = 527, shuffle=True):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.dim = dim 
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.load_dataset()
        self.train_split = train_split
        self.dataset_size = len([_ for _ in iter(self.dataset)])
        self.on_epoch_end()

    def _parse_audio_function(self, example_string):
        feature = { 
            'patches': tf.io.FixedLenFeature([], tf.string),
            'patches_shape': tf.io.FixedLenFeature(shape=(3,), dtype=tf.int64), # shape = 3 
            'label': tf.io.FixedLenFeature([self.n_classes], dtype=tf.int64), 
        }   
        feature_dict = tf.io.parse_single_example(example_string, feature)
        patches_raw = feature_dict['patches']   
        patches_shape = feature_dict['patches_shape']
        label = feature_dict['label']

        patches = tf.io.decode_raw(patches_raw, tf.float32)
        patches = tf.reshape(patches, patches_shape)
        #print(patches_shape)
        label = tf.reshape(label, (1, self.n_classes)) 
        return patches, label

    def load_dataset(self):      
        dataset = tf.data.TFRecordDataset(self.cache_dir)
        dataset = dataset.map(self._parse_audio_function)
        self.dataset = dataset.shuffle(buffer_size = 100) # 在缓冲区中随机打乱数据

    def __len__(self):
        return int(self.dataset_size * self.train_split /self.batch_size)

    def __getitem__(self, index):
        batched_train_iter = tf.compat.v1.data.make_one_shot_iterator(self.dataset)
        next_batch = batched_train_iter.get_next()
        return next_batch[0], next_batch[1]
      

    def on_epoch_end(self):
        if self.shuffle == True:
            self.dataset = self.dataset.shuffle(buffer_size = 100) # 在缓冲区中随机打乱数据
