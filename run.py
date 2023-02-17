import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from model.mobilenet_v3_small import MobileNetV3_Small, MoblieNetV3_model
from utils.datasets import DataGenerator
import utils.params as yamnet_params


print("TF version:{}".format(tf.__version__))
print("TF GPU ", )

def train(hyp, opt):
    train_cache_dir = '/home/ysr/dataset/audio/AudioSet/train.cache'
    valid_cache_dir = '/home/ysr/dataset/audio/AudioSet/valid_bak.cache'
    n_classes = 527

    model_out = './saved_models/model'
    batch_size = 1
    epochs = 100

    train_generator = DataGenerator(train_cache_dir, batch_size, n_classes = n_classes)
    valid_generator = DataGenerator(valid_cache_dir, batch_size, n_classes = n_classes)

    # Define training callbacks
    checkpoint = ModelCheckpoint(model_out+'.h5',
              monitor='val_loss', 
              verbose=1,
              save_best_only=True, 
              mode='auto')

    reducelr = ReduceLROnPlateau(monitor='val_loss', 
                factor=0.5, 
                patience=3, 
                verbose=1)

    earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=0, mode='auto')

    params = yamnet_params.Params()
    model = MobileNetV3_Small((params.patch_frames, params.mel_bands), n_classes).model(params)

    model.summary()

    # Compile model
    #optimizer = tf.keras.optimizers.Adam(lr=0.001) #0.001
    optimizer = tf.keras.optimizers.Adam(lr=0.001) #0.001
    #categorical_crossentropy binary_crossentropy
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #categorical_accuracy 
    model_history = model.fit(train_generator,
                            steps_per_epoch = len(train_generator),
                            batch_size = batch_size,
                            epochs = epochs,
                            validation_data = valid_generator,
                            validation_steps = len(valid_generator), #validation_generator
                            verbose = 1,
                            callbacks=[earlystop,checkpoint,reducelr])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    opt = parser.parse_args()
    train(None, None)




