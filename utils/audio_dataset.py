import os
import json
import numpy as np
import tensorflow as tf
import soundfile as sf
import resampy
import sys
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import argparse

sys.path.append('.')

from random import shuffle
#import features.audio_features as features_lib
import feature.audio_features as features_lib
from params import Params
import utils.tools as tools
#import utils.params as ynet_params

BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
SILENCE_LABEL = '_Silence_'
UNKNOWN_WORD_LABEL = '_Unknown_'

#设置行不限制数量
pd.set_option('display.max_rows',None)
#最后的的参数可以限制输出行的数量

#设置列不限制数量
pd.set_option('display.max_columns',None)
#最后的的参数可以限制输出列的数量

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',1000)

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
        tmp = os.listdir(train_dir + i)
        shuffle(tmp)
        for j in tmp[:round(len(tmp)*train_split)]: # loop over training samples
            if j.split('.')[-1] == file_type:
                files_train.append(train_dir + i +'/' + j)
        for j in tmp[round(len(tmp)*train_split):]: # loop over validation samples
            if j.split('.')[-1] == file_type:
                files_val.append(train_dir + i +'/' + j)
        labels[i] = cnt
    return files_train, files_val, labels


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

class ESC50DataSet():
    def __init__(self):
        print('')

    def build_dataset():
        print('')

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

'''
小数据量全部载入缓存 wav + 并且随机生成背景噪声  wav
'''
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
        files_train, files_valid, labels = get_files_and_labels(self.path + 'trian_wav/', self.file_type, self.train_split, self.wanted_label)
        self.__build_cache(files_train, labels, self.cache_dir + 'train.cache', self.params)
        self.__build_cache(files_valid, labels, self.cache_dir + 'valid.cache', self.params)
        self.__save_labels(self.path, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mine', help='audio dataset name default mine')
    parser.add_argument('--path', type=str, default='', help='audio dataset name default mine')
    parser.add_argument('--cache_dir', type=str, default=None, help='audio dataset name default mine')
    parser.add_argument('--file_type', type=str, default='wav', help='audio dataset name default mine')
    parser.add_argument('--train_split', type=float, default=0.9, help='audio dataset name default mine')
    parser.add_argument('--wanted_label', type=str, default=None, help='audio dataset name default mine')
    opt = parser.parse_args()
    param = Params()

    dataset = None

    if opt.dataset == 'mine':
        dataset = MineDataSet(param, opt.path, opt.cache_dir, opt.file_type, opt.train_split, opt.wanted_label)
    elif opt.dataset == 'audioset':
        dataset = AudioSetDataSet(param, opt.path, opt.cache_dir)
    elif opt.dataset == 'esc50':
        print('esc 50')

    dataset.build_dataset()



