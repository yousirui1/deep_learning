import os
import numpy as np
from random import shuffle
from tensorflow.python.keras.utils.data_utils import Sequence

BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
SILENCE_LABEL = '_Silence_'
UNKNOWN_WORD_LABEL = '_Unknown_'

'''
def get_files_and_labels(train_dir, typ='wav', train_split=0.9):
    
    classes = sorted(os.listdir(train_dir))
    files_train = list()
    files_val = list()
    labels = dict()
    for cnt, i in enumerate(classes): # loop over classes
        tmp = os.listdir(train_dir+i)
        shuffle(tmp)
        for j in tmp[:round(len(tmp)*train_split)]: # loop over training samples
            if j.split('.')[-1]==typ:
                files_train.append(train_dir + i +'/' + j)
        for j in tmp[round(len(tmp)*train_split):]: # loop over validation samples
            if j.split('.')[-1]==typ:
                files_val.append(train_dir + i +'/' + j)
        labels[i]=cnt
                
    return files_train, files_val, labels
'''
def get_files_and_labels(train_dir, typ='wav', train_split=0.9, wanted_label = None):  
    if not wanted_label:
        classes = sorted(os.listdir(train_dir))
    else :
        classes = [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_label.split(',') 
        #classes = wanted_label.split(',') 
    files_train = list()
    files_val = list()
    labels = dict()
 
    for cnt, i in enumerate(classes): # loop over classes
        tmp = os.listdir(train_dir+i)
        shuffle(tmp)
        for j in tmp[:round(len(tmp)*train_split)]: # loop over training samples
            if j.split('.')[-1]==typ:
                files_train.append(train_dir + i +'/' + j)
        for j in tmp[round(len(tmp)*train_split):]: # loop over validation samples
            if j.split('.')[-1]==typ:
                files_val.append(train_dir + i +'/' + j)
        labels[i] = cnt
    return files_train, files_val, labels
    
    
class DataGenerator(Sequence):
    
    'Generates YAMNet patches'
    def __init__(self, 
                 list_IDs, 
                 labels, 
                 dim = (96, 64),
                 batch_size=1, 
                 n_classes=3,
                 shuffle=True,
                 class_weights=None,
                 tflite_ouput = 1):
        'Initialization'
        self.classes = []
        self.batch_size = batch_size
        self.labels = labels
        self.dim = dim
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.class_weights = class_weights
        self.on_epoch_end()
        self.tflite_ouput = tflite_ouput
        

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



    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization   

        #X = np.empty((self.batch_size, *self.dim))
        X = []
        y = np.empty((self.batch_size, self.n_classes))
        z = np.empty((self.batch_size, 521))
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
            #if sample.shape[0] > 1:
                #sample = np.squeeze(sample[0])
             #   sample = np.squeeze(sample[np.random.choice(range(sample.shape[0]), 1)])
                
            X.append(sample)
            #X[i,] = sample
                
            if self.class_weights:
                sample_weights[i] = self.class_weights[self.labels[class_id]]
          
        self.classes.append(y.reshape(y.shape[1]))

        if self.class_weights is not None:
            return X, y, sample_weights
        else:
            if self.tflite_ouput == 1:
                return X, y
            else:
                return X, [y,z]