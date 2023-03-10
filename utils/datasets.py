import tensorflow as tf
import numpy as np
import os 
import time
#from tensorflow.python.keras.utils.data_utils import Sequence  
from tensorflow.keras.utils import Sequence

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

class DataGenerator(Sequence):
    def __init__(self, cache_dir, batch_size = 1, train_split = 1, dim = (96, 64), n_classes = 527, shuffle=True, buffer_size=None):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.dim = dim 
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.load_dataset()
        self.train_split = train_split
        self.dataset_size = len([_ for _ in iter(self.dataset)])
        if buffer_size is None:
            self.buffer_size = self.dataset_size
        else:
            self.buffer_size = buffer_size

        self.on_epoch_end()
        self.count = 0

    def _parse_audio_function(self, example_string):
        feature = { 
            'patches': tf.io.FixedLenFeature([], tf.string),
            'patches_shape': tf.io.FixedLenFeature(shape=(3,), dtype=tf.int64), # shape = 3 
            'label': tf.io.FixedLenFeature([self.n_classes], dtype=tf.int64), 
        }   
        feature_dict = tf.io.parse_single_example(example_string, feature)
        patches_raw = feature_dict['patches']   
        patches_shape = feature_dict['patches_shape']
        label = feature_dict['label']

        patches = tf.io.decode_raw(patches_raw, tf.float32)
        #patches = tf.reshape(patches, patches_shape)
        label = tf.reshape(label, (1, self.n_classes)) 
        return patches, label

    def load_dataset(self, is_audio = True):      
        dataset = tf.data.TFRecordDataset(self.cache_dir)
        self.dataset = dataset.map(self._parse_audio_function)

    def __len__(self):
        return int(self.dataset_size * self.train_split /self.batch_size)

    def __getitem__(self, index):
        start_time = time.time()
        batched_train_iter = tf.compat.v1.data.make_one_shot_iterator(self.dataset)
        one_time = time.time()
        next_batch = batched_train_iter.get_next()
        end_time = time.time()
        self.count += 1
        print("count   ", self.count, " time cost:", float(end_time - start_time) * 1000.0, "ms" , " one time cost:", float(one_time - start_time) * 1000.0, "ms")
        print('next_batch[0]', next_batch[0], 'next_batch[1]', next_batch[1])
        return next_batch[0], next_batch[1]

    def on_epoch_end(self):
        print('on_epch_end')
        if self.shuffle == True:
            self.dataset = self.dataset.shuffle(buffer_size = self.buffer_size) # 在缓冲区中随机打乱数据
