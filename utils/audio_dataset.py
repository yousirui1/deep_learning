import os
import json
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
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
import utils.tools as tools
from utils.params import Params

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


class ESC50DataSet():
    def __init__(self, params, path, cache_dir = None):
        self.params = params
        self.path = path
        self.cache_dir = cache_dir

    def __read_pd(self, path):
        meta = pd.read_csv(path +'meta/esc50.csv')
        recordings = meta.groupby('filename')['filename'].apply(lambda cat: cat.sample(1)).reset_index()['filename']
        return meta, recordings

    def __build_classes(self, class_meta, class_id):
        print('')
        #return label_index, mlb, mlb.classes_

    def __build_cache(self, wav_dir, npy_dir, meta, recordings, params):
        total_size = len(recordings)
        for index in range(len(recordings)):
            recording = recordings[index]
            recording = meta[meta.filename == recording]
            path = 'train_npy/' + recording.category.to_string(index=False).replace(" ", "") +'/'
            if os.path.exists(self.path + path) == False:
                os.makedirs(self.path + path)
            wav_file = self.path + "audio/" + recording.filename.to_string(index=False).replace(" ", "") 
            out_file = self.path + path + '16kHz_1ch_' + str(index) + ".npy"

            spectrogram, patches = log_mel_spectrogram(wav_file, params)
            
            np.save(out_file, patches)

            d = f"Scanning '{wav_file}' audio and labels... {total_size} found, {index} corrupted"
            tqdm(None, desc=d, total=total_size, initial=index)  # display cache results
            
    
    def build_dataset(self):
        meta, recordings = self.__read_pd(self.path)
        #label_index, mlb, n_classes = self.__build_classes(class_meta, class_id)
        self.__build_cache(self.path + 'train_wav/', self.path + 'train_npy/', meta, recordings, self.params)


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
        #print(labels)
        return label_index, mlb, mlb.classes_

    def __build_cache(self, wav_dir, npy_dir, file_ytid, meta,  mlb, label_index, params):
        total_size = len(file_ytid)
        
        if os.path.exists(npy_dir + 'label/') == False:
            os.makedirs(npy_dir + 'label/')
        if os.path.exists(npy_dir + 'patches/') == False:
            os.makedirs(npy_dir + 'patches/')
            
        for index, item in enumerate(file_ytid):
            label_id = []
            item = meta[meta.YTID == item]

            multi_label = item.positive_labels.to_string(index=False).split(",")
            for label in multi_label:
                label_id.append(label_index[label])

            wav_path = wav_dir + item.YTID.to_string(index=False) + '.wav'
    
            spectrogram, patches = log_mel_spectrogram(wav_path, params)
            
            np.save(npy_dir + 'label/' +  item.YTID.to_string(index=False) + '.npy', mlb.transform([label_id])[0])
            np.save(npy_dir + 'patches/' +  item.YTID.to_string(index=False) + '.npy', patches)
            
            d = f"Scanning '{wav_path}' audio and labels... {total_size} found, {index} corrupted"
            tqdm(None, desc=d, total=total_size, initial=index)  # display cache results
            
            
    def build_dataset(self):
        valid_meta, valid_id, train_meta, train_id, class_meta, class_id = self.__read_pd(self.path)
        label_index, mlb, n_classes = self.__build_classes(class_meta, class_id)

        self.__build_cache(self.path + 'train_wav/', self.path + 'train_npy/', train_id, train_meta, mlb,
                                    label_index, self.params)
        self.__build_cache(self.path + 'valid_wav/', self.path + 'valid_npy/', valid_id, valid_meta, mlb,
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
        print(len(labels))
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
        files_train, files_valid, labels = get_files_and_labels(self.path + 'train_wav/', self.file_type, self.train_split, self.wanted_label)
        self.__build_cache(files_train, labels, self.cache_dir + 'train.cache', self.params)
        self.__build_cache(files_valid, labels, self.cache_dir + 'valid.cache', self.params)
        self.__save_labels(self.path, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='esc-50', help='audio dataset name default mine')
    parser.add_argument('--path', type=str, default='', help='audio dataset name default mine')
    parser.add_argument('--cache_dir', type=str, default=None, help='audio dataset name default mine')
    parser.add_argument('--file_type', type=str, default='wav', help='audio dataset name default mine')
    parser.add_argument('--train_split', type=float, default=0.9, help='audio dataset name default mine')
    parser.add_argument('--wanted_label', type=str, default=None, help='audio dataset name default mine')
    opt = parser.parse_args()
    param = Params()

    opt.wanted_label = 'Alarm,ChainSaw,Cough,Cry,Explosion,GlassBreak,' \
                        'Knock,Laughter,Music,Scream,Siren119,Siren120,Voice'

    #if opt.dataset == 'mine':
        #opt.path = '/home/ysr/dataset/audio/mine/'
    #elif opt.dataset == 'audioset':
        #opt.path = '/home/ysr/dataset/audio/audioset/'
    #elif opt.dataset == 'esc-50':
    opt.path = '/home/ysr/dataset/audio/' + opt.dataset + '/'

    dataset = None

    if opt.dataset == 'mine':
        dataset = MineDataSet(param, opt.path, opt.cache_dir, opt.file_type, opt.train_split, opt.wanted_label)
    elif opt.dataset == 'audioset':
        dataset = AudioSetDataSet(param, opt.path, opt.cache_dir)
    elif opt.dataset == 'esc-50':
        dataset = ESC50DataSet(param, opt.path, opt.cache_dir)
        print('esc-50 ')
    else:
        print('')

    dataset.build_dataset()


