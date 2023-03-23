import soundfile as sf
import numpy as np
import resampy

#import wave
#import subprocess
#import librosa

def read_audio_file(file_path, file_type = 'wav', sample_rate = 16000):
    wav_data, sr = sf.read(file_path, dtype=np.int16) 
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype

    # int16 -> float
    waveform = wav_data / 32768.0

    if sr != sample_rate:
        waveform = resampy.resample(waveform, sr, sample_rate) 

    if len(waveform) < sample_rate:   # 1s
        waveform = np.pad(waveform, (0, sample_rate - len(waveform)), 'constant', constant_values = 0)

    # 取单通道
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    # 归一化
    waveform = np.reshape(waveform, [1, -1]).astype(np.float32)

    return waveform


"""
def read_wav(wav_path, start_time = None, end_time = None, is_play = None):
    wav_data, sr = sf.read('/home/ysr/test/test.wav', dtype=np.int32)
    # 多通道改单通道
    wav_data= np.mean(wav_data, axis = 1, dtype=wav_data.dtype)

    if start_time != None and end_time != None:
        wav_data = wav_data[sr * start_time: sr *end_time]
    elif start_time == None and end_time != None:
        wav_data = wav_data[:sr * end_time]
    elif start_time != None and end_time == None:
        wav_data = wav_data[sr * start_time:]

    if is_play not None:
        display.Audio(wav_data)


def process_backgroud(backgroud_dir, background_data):
    tmp_backgroud = os.listdir(backgroud_dir)
    print(tmp_backgroud)
    for src_path in tmp_backgroud[:round(len(tmp_backgroud))]:
        #print(backgroud_dir + src_path)
        wav_data, sr = librosa.load(backgroud_dir + src_path, sr=16000, mono=False)
        wav_data = wav_data * 32768.0
        wav_data = wav_data.astype('int16')       
        background_data.append(wav_data)
        
def add_backgroud(x, background_data):
    background_index = np.random.randint(0, len(background_data))
    if len(background_data[background_index]) > len(x):
        background_offset = np.random.randint(
            0, len(background_data[background_index]) - len(x))
        sound = x + background_data[background_index][background_offset: len(x) + background_offset] 
    else:
        sound = x
    return sound

process_backgroud('/home/ysr/dataset/audio/mine/train_wav/_Background_noise_/', background_data)

class AudioEnhancement(object):
    def __init__(self, src_path, dst_path, background_data,  backgroud_dir = None):
        self.src_path = src_path
        self.dst_path = dst_path
        self.backgroud_dir = backgroud_dir
        #self.process_backgroud()
        
        record_buf, sr = librosa.load(src_path, sr=16000, mono=False)
        record_buf = record_buf * 32768.0
        record_buf = record_buf.astype('int16')
    
        wav_data = self.time_shift(record_buf, len(record_buf) / np.random.randint(1,5))
        #wav_data = self.add_noise1(wav_data)
        #wav_data = self.add_backgroud(wav_data)
        wav_data = add_backgroud(wav_data, background_data)
        #return wav_data
        wf = wave.open('temp.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(wav_data.tobytes())
        wf.close()
        
        self.modify_db_wav('temp.wav', dst_path)

    
    def time_shift(self, x, shift):
        # shift：移动的长度
        return np.roll(x, int(shift))   # 循环移动
    
    def time_stretch(self, x, rate):
        # rate：拉伸的尺寸，
        # rate > 1 加快速度
        # rate < 1 放慢速度
        return librosa.effects.time_stretch(x, rate)

    def pitch_shifting(x, sr, n_steps, bins_per_octave=12):
        # sr: 音频采样率
        # n_steps: 要移动多少步
        # bins_per_octave: 每个八度音阶(半音)多少步
        return librosa.effects.pitch_shift(x, sr, n_steps, bins_per_octave=bins_per_octave)

    def add_noise2(x, snr):
        # snr：生成的语音信噪比
        P_signal = np.sum(abs(x) ** 2) / len(x)  # 信号功率
        P_noise = P_signal / 10 ** (snr / 10.0)  # 噪声功率
        return x + np.random.randn(len(x)) * np.sqrt(P_noise)

    def add_noise1(x, w=0.004):
        # w：噪声因子
        output = x + w * np.random.normal(loc=0, scale=1, size=len(x))
        return output
    
    def process_backgroud(self):
        tmp_backgroud = os.listdir(self.backgroud_dir)
        self.background_data = []
        for src_path in tmp[:round(len(tmp_backgroud))]:
            wav_data, sr = librosa.load(self.backgroud_dir + src_path, sr=16000, mono=False)
            wav_data = wav_data * 32768.0
            wav_data = wav_data.astype('int16')       
            self.background_data.append(wav_data)
            
    def add_backgroud(self, x):
        background_index = np.random.randint(0, len(self.background_data))
        background_offset = np.random.randint(
            0, len(self.background_data[background_index]) - len(sound1))
        
        sound = x + self.background_data[background_index][background_offset: len(sound1) + background_offset] 
        return sound
    
    def mix(sound1, sound2, r, fs):
        gain1 = np.max(compute_gain(sound1, fs))  # Decibel
        gain2 = np.max(compute_gain(sound2, fs))
        t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
        sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2)) 
        return sound
    
    def modify_db_wav(self, src_path, dst_path):
        subprocess.call(["ffmpeg", "-i", src_path,  "-ar", "16000", "-ac", "1",  "-filter:a", "volume={}dB".format(np.random.randint(-3,8)), dst_path])
"""
