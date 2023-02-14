import tensorflow as tf
import os 
os.sys.path.append('./audioset/yamnet')
import params as yamnet_params
import features as features_lib
import soundfile as sf
import numpy as np
import resampy
from pathlib import Path
import glob
from tqdm import tqdm
from tensorflow.python.keras.utils.data_utils import Sequence
from random import shuffle

audio_formats = ['wav', 'aac', 'pcm']

# PCM
class LoadSounds:
    print("")
    #def __initfor k in range(ts_len):
    #ts_file = ts_list[k]
    #file_name = 



# RGB
class LoadImages:
    print("")
    #def __init__(self, path, img_size = 640, stride=32):
        #p = str(Path(path).absolute())      # os-agnostic absolute path
        #if '*' in p:
            #ZZ


def _bytes_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _numpy_int32_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.reshape(-1)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(np.float32).tostring()]))


def _numpy_float32_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.reshape(-1)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(np.float32).tostring()]))

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

def _shape_feature(value):
    # Returns a int64_list from a bool / enum / int /uint 
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def log_mel_spectrogram(audio_file, param):
    wav_data, sr = sf.read(audio_file, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype

    if len(wav_data) < param.sample_rate:
        # 补0 to do  mask
        wav_data = np.pad(wav_data, (0, param.sample_rate - len(wav_data)), 'constant', constant_values = 0)

    waveform = wav_data / 32768.0

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)  # 2 维转 3维 

    if sr != param.sample_rate:
        waveform = resampy.resample(waveform, sr, param.sample_rate)

    waveform = np.reshape(waveform, [1, -1]).astype(np.float32)
    return features_lib.waveform_to_log_mel_spectrogram_patches(tf.squeeze(waveform, axis=0), param)


class LoadAudioSet():
    def __init__(self, path, params, single_cls=False, rect=False, prefix=''):
        self.path = path
        self.params = params
        self.rect = rect

        #try:
        #    f = [] #audio files
        #    for p in path if isinstance(path, list) else [path]:
        #        p = Path(p) # os-agnostic
        #        if p.is_dir():
        #            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
        #        elif p.is_file(): # file  .txt
        #            if sorted([p.replace('/', os.sep) for p in f if x.split('.')[-1].lower() in audio_formats]):
        #                with open(p, 'r') as t:
        #                    t = t.read().strip().splitlines()
        #                    parent = str(p.parent) + os.sep
        #                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
        #        else:
        #            raise Exception(f'{prefix}{p} does not exist')

        #    self.audio_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in audio_formats])
        #    print(len(self.audio_files))
        #    assert self.audio_files, f'{prefix} No audio found'
        #except Exception as e:
        #    raise Exception(f'{prefix}Error loading data from {path}: {e}\n')
        
        # Check cache 
        self.label_files = 'valid'
        cache_path = (p if p.is_dir() else Path(self.label_files)).with_suffix('.cache') # to do

        if cache_path.is_file():
            cache, exists = self.load_cache(str(cache_path)), True
        else:
            cache, exists  = self.cache_data(str(cache_path)), False

        # Displya cache
        #cache('result') # found, missing, empty, corrupted, total
        #print()
        if exists:
            print('')
            #d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            #tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        
        if single_cls:
            print('single_cls') # 单标签  to do 
            #for x in 

        # Recangular Training
        if self.rect:
            print('rect !!')  #to do

    def audio_example(self, patches, label): # tf record example
        feature = {
            'patches': _numpy_float32_feature(patches),
            'patches_shape': _shape_feature(patches.shape),
            'label': _numpy_int32_feature(label),
            'label_shape': _shape_feature(label.shape),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)) 

    def _parse_audio_function(self,example_proto):
        feature = {
            'patches': tf.io.FixedLenFeature([], tf.string),
            'patches_shape': tf.io.FixedLenFeature(shape=(3,), dtype=tf.int64), # shape = 3 
            'label': tf.io.FixedLenFeature([], tf.string),
            'label_shape': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64), #shape (1,)
        }
        return tf.io.parse_single_example(example_proto, feature)
        
    def load_cache(self, cache_path):
        raw_audio_dataset = tf.data.TFRecordDataset(cache_path)
        audio_cache = raw_audio_dataset.map(self._parse_audio_function)
        return audio_cache

    def cache_data(self, cache_path):
        param = yamnet_params.Params()
        count = 0
        with tf.io.TFRecordWriter(cache_path) as writer:
            for audio_file in self.audio_files:
                #d = f"total len {len(self.audio_files)} len {count}"
                print(f"total len {len(self.audio_files)} len {count}")
                #print(d)
                count += 1
                spectrogram, patches = log_mel_spectrogram(audio_file, param)
                example = self.audio_example(patches.numpy(), 0)
                writer.write(example.SerializeToString())
        return self.load_cache(cache_path)

    def __len__(self):
        return self.cache.size()

class DataGenerator(Sequence):
    def __init__(self, dataset, batch_size=1, dim = (96, 64), shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    #def __len__(self):
    #    return int(np.floor(self.dataset.__len__) / self.batch_size)

    #def __getitem(self, index):
    #    index[;]

    #    if self.class_weights:
    #        return
    #    else:
    #        x,y = self.__data_generation()
    #        return x, y

    #def __data_generation(self, list_id_temp):
    #    print('');

    def on_epoch_end(self):
        self.indexes = np.arange(self.dataset.__len__())
        if self.shuffle == True:
            np.random.shuffle(self.indexes)






