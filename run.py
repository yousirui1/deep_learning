import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from model.mobilenet_v3_small import MobileNetV3_Small, MoblieNetV3_model
from utils.datasets import DataGenerator
import utils.params as yamnet_params

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("TF GPU: {}".format(tf.test.gpu_device_name()), " enable: {}".format(tf.test.is_gpu_available()))
print("TF version:{}".format(tf.__version__))

def train(hyp, opt):
    train_cache_dir = '/home/ysr/dataset/audio/audioset/train.cache'
    valid_cache_dir = '/home/ysr/dataset/audio/audioset/valid.cache'
    n_classes = 527

    model_out = './saved_models/model'
    batch_size = 1
    epochs = 100

    train_generator = DataGenerator(train_cache_dir, batch_size, n_classes = n_classes)
    valid_generator = DataGenerator(valid_cache_dir, batch_size, train_split = 0.3,  n_classes = n_classes)

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
    if opt.weights and os.path.exists(opt.weights):
        print(opt.weights)
        model.load_weights(opt.weights)

    model.summary()

    # Compile model
    #optimizer = tf.keras.optimizers.Adam(lr=0.001) #0.001
    optimizer = tf.keras.optimizers.SGD(lr=0.001) #0.001
    #categorical_crossentropy binary_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #categorical_accuracy 
    #model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #categorical_accuracy 
    model_history = model.fit(train_generator,
                            steps_per_epoch = len(train_generator),
                            batch_size = batch_size,
                            epochs = epochs,
                            validation_data = valid_generator,
                            validation_steps = len(valid_generator), #validation_generator
                            verbose = 1,
                            callbacks=[checkpoint,reducelr])
#                            callbacks=[earlystop,checkpoint,reducelr])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    opt = parser.parse_args()
    train(None, opt)




