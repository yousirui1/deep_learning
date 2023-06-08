import torch
import torchaudio
from torchaudio.transforms import Vol

def _wav2fbank(filename, filename2 = None, backgroud = None, target_length=998,num_mel_bins=126):
    # mixup
    if filename2 == None:
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
    # mixup
    else:
        waveform1, sr = torchaudio.load(filename)
        waveform2, _ = torchaudio.load(filename2)

        waveform1 = waveform1 - waveform1.mean()
        waveform2 = waveform2 - waveform2.mean()

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
        #mix_lambda = random.random()
        # sample lambda from beta distribtion
        mix_lambda = np.random.beta(10, 10) 

        mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
        waveform = mix_waveform - mix_waveform.mean()

    if backgroud != None:
        backgroud_db = 50  # 设置目标分贝值
        vol_transform = Vol(volume=backgroud_db)

        # 将音频应用到 Vol 变换
        backgroud_waveform = vol_transform(torchaudio.load(backgroud))
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
        []
        waveform = waveform + backgroud_waveform

    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, 
                use_energy=False,window_type='hanning', num_mel_bins=melbins, dither=0.0, 
                frame_shift=10)

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


if __name__ == '__main__':
    fbank, _ = wav2fbank(1056, 128, "/home/ysr/project/dataset/audio/dcase2020_task2/train/fan/test/normal_id_06_00000099.wav")
    print(fbank)
