import os
from torch.utils.data import Dataset

      if use_background or sample['label'] == SILENCE_LABEL:
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]
        if len(background_samples) <= model_settings['desired_samples']:
          raise ValueError(
              'Background sample is too short! Need more than %d'
              ' samples but only %d were found' %
              (model_settings['desired_samples'], len(background_samples)))
        background_offset = np.random.randint(
            0, len(background_samples) - model_settings['desired_samples'])
        background_clipped = background_samples[background_offset:(
            background_offset + desired_samples)]
        background_reshaped = background_clipped.reshape([desired_samples, 1])
        if sample['label'] == SILENCE_LABEL:
          background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < background_frequency:
          background_volume = np.random.uniform(0, background_volume_range)
        else:
          background_volume = 0
      else:
        background_reshaped = np.zeros([desired_samples, 1])
        background_volume = 0
      input_dict[self.background_data_placeholder_] = background_reshaped
      input_dict[self.background_volume_placeholder_] = background_volume

class MineDataset(Dataset):
    def __init__(self, root_dir, train=True, classes=None,  audio_conf, transform=None, 
                    file_type='wav'):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.data = []
        self.audio_conf = audio_conf
        self.backgroud = audio_conf.get('backgroud')
        self.melbins = audio_conf.get('num_mel_bins')
        self.freqm = audio_conf.get('freqm')
        self.timem = audio_conf.get('timem')
        self.mixup = audio_conf.get('mixup')
        self.norm_mean = audio_conf.get('mean')
        self.norm_std = audio_conf.get('std')
        self.skip_norm = audio_conf.get('skip_norm')
        self.noise = audio_conf.get('noise')

        if self.train == True:
            print('dataset use train')
        else
            print('dataset use eval')

        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization'
                  'stats')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input'.format(self.norm_mean, self.norm_std))

        if self.noise == True:
            print('now use noise augmentation')

        if self.backgroud != 0:
            print('now use backgroud augmentation')

        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                files = os.listdir(class_dir)
                for file_name in files:
                    file_path = os.path.join(class_dir, file_name)
                    self.data.append((file_path, class_name))
       
        self.label_map = {class_name: i for i, class_name in enumerate(classes)}
        self.label_num = len(classes))
        print('label len ', len(classes))

    def __prepare_background_data(self):
        self.backgroud_data = []
        for item in os.listdir(root_dir + '_backgroud_noise_/'):
            if item.split('.')[-1] == self.file_type:
                self.backgroud_data.append(item)
                print(item)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        use_backgroud_noise = False

        # backgroud
        if random.random() < self.backgroud:
            use_backgroud = True
        
        # mixup
        if random.random() < self.mixup:
            datum, label_name = self.data[index]
            
            mix_sample_idx = random.randint(0, len(self.data) -1)
            mix_datum, mix_label_name = self.data[mix_sample_idx]

            if use_backgroud == True:
                backgroud_sample_idx = random.randint(0, len(self.backgroud_data) -1)
                backgroud_datum = self.backgroud_data[backgroud_sample_idx]
                fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'], 
                                        backgroud_datum['wav'])
            else:
                fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])

            #initialize the label
            label_indices = np.zeros(self.label_num)

            label_indices[self.label_map[label_name]] += mix_lambda
            label_indices[self.label_map[mix_label_name]] += (1.0 - mix_lambda)
            label_indices = torch.FloatTensor(label_indices)

        else:
            datum, label_name = self.data[index]
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'], 
                                        backgroud_datum['wav'])
            label_indices = np.zeros(self.label_num)
            label_indices[self.label_map[label_name]] += mix_lambda
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        freqm = torchaudio.transforms.TimeMasking(self.timem)
        freqm = torchaudio.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)

        # squeeze back
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        
        features, class_name = self.data[index]
        label = self._get_label(class_name)
        return features, label

    def _get_label(self, class_name):
        # 这里根据需要定义标签的映射关系，例如将类别名转换为数字标签
        return self.label_map[class_name]


