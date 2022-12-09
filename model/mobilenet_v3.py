from tensorflow import keras
from keras.models import Input, Model
from model.mobilenet_v3_small import MobileNetV3_Small
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class MobileNetV3(Model):
    def __init__(self, shape, n_class, alpha=1.0):
        super().__init__()
        self.shape = shape
        self.n_class = n_class
        self.build_model()

    def build_model(self):
       	self.model = MobileNetV3_Small(self.shape, self.n_class).build()

    def summary(self):
        return self.model.summary()

    def compile_(self, lr = 0.0001, loss = "categorical_crossentropy", opt = "adam"):
        if opt == "adam":
            opt = keras.optimizers.Adam(learning_rate = lr)
        else:
            opt = opt = keras.optimizers.SGD(learning_rate = lr)
        self.model.compile(loss = loss, optimizer = opt, metrics=['accuracy'])

    ## adding some model checkpoints to ensure the best values will be saved
    ## and early stopping to prevent the model from running in vain
    def callbacks(self, **kwargs):
    	self.checkpoint = ModelCheckpoint(filepath = kwargs.get("filename"),
                             monitor=kwargs.get("monitor"), 
                             verbose=1,
                             save_best_only=True, 
                             mode='auto')

    	self.reducelr = ReduceLROnPlateau(
                              monitor=kwargs.get("monitor"), 
                              factor=0.5, 
                              patience=kwargs.get("patience"),           
                              verbose=1)

        #self.earlystop = EarlyStopping(monitor = kwargs.get("monitor"),
        #                                patience = kwargs.get("patience"))

    ## model fit
    def train(self, x, y, x_val = None, y_val = None, 
              n_epochs = 15, batch_size = 32, 
              verbose = 1, callbacks = None):
        if x_val is None:
            self.model.fit(x, y, validation_split = 0.2,
                          epochs = n_epochs, verbose = verbose,
                          batch_size = batch_size)
        else:
            self.model.fit(x, y, validation_data = (x_val, y_val),
                          epochs = n_epochs, verbose = verbose,
                          batch_size = batch_size, 
                          callbacks = [self.checkpoint, self.reducelr])
    
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



def test():
    mobilenet_v3 = MobileNetV3((96, 64), 10)
    mobilenet_v3.summary()
