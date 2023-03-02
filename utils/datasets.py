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
from random import shuffle

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

class DataGenerator(Sequence):
    def __init__(self, cache_dir, batch_size = 1, train_split = 1, dim = (96, 64), n_classes = 527, shuffle=True):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.dim = dim 
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.load_dataset()
        self.train_split = train_split
        self.dataset_size = len([_ for _ in iter(self.dataset)])
        self.on_epoch_end()

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
        patches = tf.reshape(patches, patches_shape)
        #print(patches_shape)
        label = tf.reshape(label, (1, self.n_classes)) 
        return patches, label

    def load_dataset(self):      
        dataset = tf.data.TFRecordDataset(self.cache_dir)
        dataset = dataset.map(self._parse_audio_function)
        self.dataset = dataset.shuffle(buffer_size = 100) # 在缓冲区中随机打乱数据

    def __len__(self):
        return int(self.dataset_size * self.train_split /self.batch_size)

    def __getitem__(self, index):
        batched_train_iter = tf.compat.v1.data.make_one_shot_iterator(self.dataset)
        next_batch = batched_train_iter.get_next()
        return next_batch[0], next_batch[1]
      

    def on_epoch_end(self):
        if self.shuffle == True:
            self.dataset = self.dataset.shuffle(buffer_size = 100) # 在缓冲区中随机打乱数据
