
class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print('------------- the {:s} dataloader-----------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask {:d} freq, {:d}time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))

        self.mixup = self.audio_conf.get('mixup')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you 
        # set it as True ONLY when you are getting the normalization stats
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats)')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to nromaliza the input'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        else:
            wavform1, sr = torchaudio.load(filename)
            wavform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveforme1.mean()
            waveform2 = waveform2 - waveforme2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            # mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat = True, sample_frequency=sr, use_energy=False,
                    window_type = 'hanning', num_mel_bins = self.melbins, dither = 0.0, frame_shift = 10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        """
        if random.random() < self.mixup:
            datum = self.data[index]
            mix_sample_idx = random.randint(0, len(self.data) - 1)
            mix_datum = self.data[mix_sample_idx]

            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label onehot 
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices = [int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices = [int(self.index_dict[label_str])] += (1.0 - mix_lambda)
        # if not do mixup
        else:
            datum = self.data[index]

        if self.noise == True:
            fbank = fbank + torch.rand() * np.random.rand() / 10
            fbank = torch.roll()

            

