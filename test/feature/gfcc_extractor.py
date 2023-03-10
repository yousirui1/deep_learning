# coding = utf-8
import numpy as np
#from scipy.io import wavfile
import wave
from feature_extractor import cochleagram_extractor
from matplotlib import  pyplot as plt

def load_wave_data(path):
    wave_file = wave.open(path, 'rb')
    params = wave_file.getparams()
    channel, samplewidth, samplerate, frames = params[:4]
    #print('channel %d samplewidth %d samplerate %d frames %d', 
          #channel, samplewidth, samplerate, frames)
    str_data = wave_file.readframes(frames)
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))
    return wave_data

def gfcc_extractor(cochleagram, gf_channel, cc_channels):
    dctcoef = np.zeros((cc_channels, gf_channel))
    for i in range(cc_channels):
        n = np.linspace(0, gf_channel-1, gf_channel)
        dctcoef[i, :] = np.cos((2 * n + 1) * i * np.pi / (2 * gf_channel))
    plt.figure()
    plt.imshow(dctcoef)
    plt.show()
    return np.matmul(dctcoef, cochleagram)


if __name__ == '__main__':
    # wav_data, wav_header = read_sphere_wav(u"clean.wav")

    #sr, wav_data = wavfile.read(u"clean.wav")
    wav_data = load_wave_data("../audio_rknn/gru_ok/model/recoard_baby.wav")
    sr = 16000
    cochlea = cochleagram_extractor(wav_data, sr, 320, 160, 64, 'hanning')
    gfcc = gfcc_extractor(cochlea, 64, 31)

    #plt.figure(figsize=(10,8))
    #plt.subplot(211)
    #plt.imshow(np.flipud(cochlea))

    #plt.subplot(212)
    #plt.imshow(np.flipud(gfcc))
    #plt.show()

    #plt.figure(figsize=(10, 8))
    #plt.plot(gfcc[0,:])
    #plt.show()
