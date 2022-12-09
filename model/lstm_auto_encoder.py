import tensorflow as tf
from tensorflow import keras
from keras.models import Input, Model
from keras.layers import Dense, LSTM
from keras.layers import RepeatVector, TimeDistributed
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping


class LSTMAutoencoder(Model):
    ## the class is initiated with the number of dimensions and 
    ## timesteps (the 2nd dimension in the 3 dimensions) 
    ## and the number of dimensions to which we want the encoder 
    ## to represent the data (bottleneck)
    def __init__(self, n_dims, n_timesteps = 1, 
                 n_bottleneck = 8, bottleneck_name = "bottleneck"):
        super().__init__()
        self.bottleneck_name = bottleneck_name
        self.build_model(n_dims, n_timesteps, n_bottleneck)
        
    
    ## each of the encoder and decoder will have two layers 
    ## with hard coded parameters for simplicity
    def build_model(self, n_dims, n_timesteps, n_bottleneck):
        self.inputs = Input(shape = (n_timesteps, n_dims))
        e = LSTM(16, activation = "relu", return_sequences = True)(self.inputs)
        ## code layer or compressed form of data produced by the autoencoder
        self.bottleneck = LSTM(n_bottleneck, activation = "relu", 
                               return_sequences = False, 
                               name = self.bottleneck_name)(e)
        e = RepeatVector(n_timesteps)(self.bottleneck)
        decoder = LSTM(n_bottleneck, activation = "relu", 
                       return_sequences = True)(e)
        decoder = LSTM(16, activation = "relu", return_sequences = True)(decoder)
        self.outputs = TimeDistributed(Dense(n_dims))(decoder)
        self.model = Model(inputs = self.inputs, outputs = self.outputs)
        
    
    ## model summary
    def summary(self):
        return self.model.summary()
    
    
    ## compiling the model with adam optimizer and mean squared error loss
    def compile_(self, lr = 0.0003, loss = "mse", opt = "adam"):
        if opt == "adam":
            opt = keras.optimizers.Adam(learning_rate = lr)
        else:
            opt = keras.optimizers.SGD(learning_rate = lr)
        self.model.compile(loss = loss, optimizer = opt)
    

    
    ## adding some model checkpoints to ensure the best values will be saved
    ## and early stopping to prevent the model from running in vain
    def callbacks(self, **kwargs):
        self.mc = ModelCheckpoint(filepath = kwargs.get("filename"), 
                                  save_best_only = True, verbose = 0)
        
        self.es = EarlyStopping(monitor = kwargs.get("monitor"),
                                patience = kwargs.get("patience"))
        
    
    ## model fit
    def train(self, x, y, x_val = None, y_val = None, 
              n_epochs = 15, batch_size = 32, 
              verbose = 1, callbacks = None):
        if x_val is not None:
            self.model.fit(x, y, validation_split = 0.2,
                          epochs = n_epochs, verbose = verbose,
                          batch_size = batch_size)
        else:
            self.model.fit(x, y, validation_data = (x_val, y_val),
                          epochs = n_epochs, verbose = verbose,
                          batch_size = batch_size, 
                          callbacks = [self.mc, self.es])
    
    
    ## reconstruct the new data 
    ## should be with lower error for negative values than
    ## the error with the positive ones
    def predict(self, xtest):
        return self.model.predict(xtest)
    
    
    ## after investigating the error differences, we set the threshold
    def set_threshold(self, t):
        self.threshold = t
        
    ## after setting the threshold, we can now predict the classes from the
    ## reconstructed data
    def predict_class(self, x_test, predicted):
        mse = mse_3d(x_test, predicted)
        return 1 * (mse > self.threshold)
    
    
    ## in case we are interested in extracting the (bottleneck) the low-dimensional 
    ## representation of the data
    def encoder(self, x):
        self.encoder_layer = Model(inputs = self.inputs, outputs = self.bottleneck)
        return self.encoder_layer.predict(x)


def test():
    inputs_dim = 64
    lstm = LSTMAutoencoder(inputs_dim)
    lstm.model.summary()
    lstm.callbacks(filename = "../model/lstm_autoenc.h5", patience = 5, monitor = "val_loss")
    lstm.compile_(opt='adam')
    #lstm.train();

test()
