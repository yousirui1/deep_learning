import tensorflow as tf
import numpy as np
import os
os.sys.path.append('./audioset/yamnet')

#import params as yamnet_params
#import features as features_lib
#import soundfile as sf
#import numpy as np
#import resampy
from pathlib import Path
import glob
from tqdm import tqdm
#from tensorflow.python.keras.utils.data_utils import Sequence  
from tensorflow.keras.utils import Sequence


audio_formats = ['wav', 'aac', 'pcm']
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
SILENCE_LABEL = '_Silence_'
UNKNOWN_WORD_LABEL = '_Unknown_'


def get_files_and_labels(train_dir, file_type = 'wav', train_split = 0.9, wanted_label = None):
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
            if j.split('.')[-1] == file_type:
                files_train.append(train_dir + i +'/' + j)
        for j in tmp[round(len(tmp)*train_split):]: # loop over validation samples
            if j.split('.')[-1] == file_type:
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
            
            X[i,] = sample

            if self.class_weights:
                sample_weights[i] = self.class_weights[self.labels[class_id]]

        self.classes.append(y.reshape(y.shape[1]))

        if self.class_weights is not None:
            return X, y, sample_weights
        else:
            return X, y


import tensorflow as tf
import os
os.sys.path.append('/home/ysr/project/ai/yamnet-transfer-learning/audioset/yamnet')
import params as yamnet_params

def _bytes_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _numpy_float32_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.reshape(-1)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(np.float32).tostring()]))

def _numpy_int32_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.reshape(-1)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(np.int32).tostring()]))

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

def _int64_list_feature(value):
    # Returns a int64_list from a bool / enum / int /uint 
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _shape_feature(value):
    # Returns a int64_list from a bool / enum / int /uint 
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

import os
import subprocess
import numpy as np
import pandas as pd
import wavefile
import wave
from IPython import display
from sklearn.preprocessing import MultiLabelBinarizer

import contextlib
import wave
import librosa
#import webrtcvad
import os
import sys
import collections
import numpy as np
import struct
import csv
import json
import soundfile as sf
import resampy
import features as features_lib
from tqdm import tqdm

#设置行不限制数量
pd.set_option('display.max_rows',None)
#最后的的参数可以限制输出行的数量

#设置列不限制数量
pd.set_option('display.max_columns',None)
#最后的的参数可以限制输出列的数量

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',1000)

def log_mel_spectrogram(audio_file, param, start_time = None, end_time = None):
    wav_data, sr = sf.read(audio_file, dtype=np.int16)
    
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype

    if len(wav_data) < sr:
        # 补0 to do  mask
        wav_data = np.pad(wav_data, (0, sr - len(wav_data)), 'constant', constant_values = 0)

    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1, dtype=wav_data.dtype)  # 多通道变单通道
    
   
    # 音频切割
    if start_time != None and end_time != None:
        wav_data = wav_data[(int)(sr * start_time): int(sr *end_time)]
    elif start_time == None and end_time != None:
        wav_data = wav_data[:(int)(sr * end_time)]
    elif start_time != None and end_time == None:
        print(start_time)
        wav_data = wav_data[(int)(sr * start_time):]
    
    waveform = wav_data / 32768.0
        
    if sr != param.sample_rate:
        waveform = resampy.resample(waveform, sr, param.sample_rate)

    waveform = np.reshape(waveform, [1, -1]).astype(np.float32)
    return features_lib.waveform_to_log_mel_spectrogram_patches(tf.squeeze(waveform, axis=0), param)

def audio_example(patches, label): # tf record example
    feature = { 
        'patches': _numpy_float32_feature(patches),
        'patches_shape': _shape_feature(patches.shape),
        'label': _int64_list_feature(label),
        #'label_shape': _shape_feature(label.shape),
    }   
    return tf.train.Example(features=tf.train.Features(feature=feature)) 


def AudioSet(param, cache_path):
    valids_meta = pd.read_csv('/home/ysr/dataset/audio/audioset/valid.csv')
    valids_id = valids_meta.groupby('YTID')['YTID'].apply(lambda cat: cat.sample(1)).reset_index()['YTID']
    class_meta = pd.read_csv('/home/ysr/dataset/audio/audioset/class_labels_indices.csv')
    class_id = class_meta.groupby('index')['index'].apply(lambda cat: cat.sample(1)).reset_index()['index']
    label_index = {}
    labels = []
    for index in range(len(class_id)):
        item = []
        clas = class_id[index]
        clas = class_meta[class_meta.index == clas]
        label_index[clas.mid.to_string(index=False)] = clas.index[0]
        item.append(clas.index[0])
        labels.append(item)
    

    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    print(mlb.classes_)
        
    nf = len(valids_id)
    nc = 0
    n = 0
    print(nf)
    with tf.io.TFRecordWriter(cache_path) as writer:
        for index in range(len(valids_id)):
            label_id = []
            valid = valids_id[index]
            valid = valids_meta[valids_meta.YTID == valid]
        
            #print(f'start{valid.start_seconds.to_string(index=False)} end {valid.end_seconds.to_string(index=False)} positive_labels {valid.positive_labels.to_string(index=False)}')
            
            label_mid = valid.positive_labels.to_string(index=False).split(",")
        
            for i in label_mid :
            #    print(label_index)
                label_id.append(label_index[i])
            #print('/home/ysr/mnt/audio/audio_set/valid_wav/' + valid.YTID.to_string(index=False) + '.wav')
            wav_path = '/home/ysr/dataset/audio/audioset/valid_wav/' + valid.YTID.to_string(index=False) + '.wav'
            spectrogram, patches = log_mel_spectrogram(wav_path, param)
                                                    # 当前数据已经处理好了，不需要切割
                                                   #param, float(valid.start_seconds.to_string(index=False)), float(valid.end_seconds.to_string(index=False)))
            
            #y = np.zeros((1, 10), dtype=np.int64)   # batch_size
            #y[0, [3]] = 1
            #print(len(label_id))
            #print(type(mlb.transform([label_id])))
            #print(mlb.transform([label_id]))
            #print(mlb.transform([label_id]).type)
            #print(mlb.transform([label_id]).shape)
            
            example = audio_example(patches.numpy(), mlb.transform([label_id])[0])

            #print(label_id.shape)
            writer.write(example.SerializeToString())
            nc += 1
            if nc == 1:
                break
            #print(f'count: {count}')
            d = f"Scanning '{wav_path}' audio and labels... {nf} found, {nc} corrupted"
            tqdm(None, desc=d, total=nf, initial=nc)  # display cache results


param = yamnet_params.Params()
AudioSet(param, '/home/ysr/dataset/audio/audioset/valid_test1111.cache')

def _parse_audio_function(example_proto):
    feature = {
        'patches': tf.io.FixedLenFeature([], tf.string),
        'patches_shape': tf.io.FixedLenFeature(shape=(3,), dtype=tf.int64), # shape = 3 
        'label': tf.io.FixedLenFeature([527], dtype=tf.int64), # shape = 3 
        #'label_shape': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
        #'label': tf.io.FixedLenFeature([], tf.int64), # shape = 3 
        #'label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature)

def load_cache(cache_path):
    raw_audio_dataset = tf.data.TFRecordDataset(cache_path)
    audio_cache = raw_audio_dataset.map(_parse_audio_function)
    return audio_cache

cache = load_cache('/home/ysr/mnt/audio/audio_set/valid.cache')
#print(cache.size())

for audio_features in cache:
    patches_raw = audio_features['patches']
    patches_shape = audio_features['patches_shape']
    label_raw = audio_features['label']
    #label_shape = audio_features['label_shape']
    patches = tf.io.decode_raw(patches_raw, tf.float32)
    patches = tf.reshape(patches, patches_shape)
    print(label_raw)
    #label = tf.io.decode_raw(label_raw, tf.int32)
    #label = tf.reshape(label, label_shape)

    #print(patches)
    #print('1111111111')
    #print(label)
