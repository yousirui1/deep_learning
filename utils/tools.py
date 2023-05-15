import numpy as np
import os
from random import shuffle
import time

BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
SILENCE_LABEL = '_silence_'
UNKNOWN_WORD_LABEL = '_unknown_'

def log(msg):
    print('\n===========\n{} | {} \n ============\n'.format(time.asctime(),msg), flush=True)


def get_files_and_labels(train_dir, file_type = 'wav', train_split = 1, wanted_label = None, single_cls = True):
    files_train = list()
    files_val = list()
    labels = dict()

    if single_cls :
        if not wanted_label:
            classes = sorted(os.listdir(train_dir))
        else :
            classes = [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_label.split(',') 
 
        for cnt, i in enumerate(classes): # loop over classes
            tmp = os.listdir(train_dir + i)
            shuffle(tmp)
            for j in tmp[:round(len(tmp)*train_split)]: # loop over training samples
                if j.split('.')[-1] == file_type:
                    files_train.append(train_dir + i +'/' + j)
            for j in tmp[round(len(tmp)*train_split):]: # loop over validation samples
                if j.split('.')[-1] == file_type:
                    files_val.append(train_dir + i +'/' + j)
            labels[i] = cnt 
    else:
        tmp = os.listdir(train_dir)
        shuffle(tmp)
        for j in tmp[:round(len(tmp)*train_split)]: # loop over training samples
            if j.split('.')[-1] == file_type:
                files_train.append(train_dir  +'/' + j)
        for j in tmp[round(len(tmp)*train_split):]: # loop over validation samples
            if j.split('.')[-1] == file_type:
                files_val.append(train_dir  +'/' + j)

    return files_train, files_val, labels


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

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

