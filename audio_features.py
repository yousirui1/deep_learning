import torch
import torchaudio


def wav2fbank(target_length, mel_bins, filename, filename2 = None):
    # 
    if filename2 == None:
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
    else: # mixup
        mix_lambda = None

    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat = True,
             sample_frequency = sr, use_energy=False, window_type = 'hanning', 
             num_mel_bins = mel_bins, dither = 0.0, frame_shift = 10)
    
    n_frames = fbank.shape[0]
    print(fbank.shape)

    p = target_length - n_frames

    # cut and pad 
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0 : target_length, :]

    if filename2 == None:
        return fbank, 0
    else:
        return fbank, mix_lambda

if __name__ == '__main__':
    fbank, _ = wav2fbank(1056, 128, "/home/ysr/project/dataset/audio/dcase2020_task2/train/fan/test/normal_id_06_00000099.wav")
    print(fbank)
