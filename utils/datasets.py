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
    
    'Generates YAMNet patches'
    def __init__(self, 
                 list_IDs, 
                 labels, 
                 dim = (96, 64),
                 batch_size=1, 
                 n_classes=3,
                 use_background = 0,
                 single_cls = True,
                 shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.dim = dim 
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.single_cls = single_cls
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
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization   
        x = []
        y = []
        
        for i, ID in enumerate(list_IDs_temp):
            label_id = ID.replace('patches', 'label')
            patches = np.load(ID)
            label = np.load(label_id)
            x.append(patches)
            y.append(label)

        return tf.cast(x[0], dtype=tf.float32), tf.cast(y, dtype=tf.int32)

class DataGenerator_1(Sequence):
    
    'Generates YAMNet patches'
    def __init__(self, 
                 list_IDs, 
                 labels, 
                 dim = (96, 64),
                 batch_size=1, 
                 n_classes=3,
                 use_background = 0,
                 shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.dim = dim 
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
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
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

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

        return tf.cast(X, dtype=tf.float32), tf.cast(y, dtype=tf.int32)

