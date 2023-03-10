import os
import json
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from model.mobilenet_v3 import MobileNetV3Small
#from model.mobilenet_v3_small import MobileNetV3_Small, MoblieNetV3_model
from utils.datasets import DataGenerator
from utils.params import Params
from audioset.yamnet.yamnet import yamnet_model

print("TF GPU: {}".format(tf.test.gpu_device_name()), " enable: {}".format(tf.test.is_gpu_available()))
print("TF version:{}".format(tf.__version__))

def build_mode(opt, n_classes):
    params = Params()
    model = None

    print('model_name: ', opt.model_name)
    if opt.model_name == 'yamnet':
        yamnet = yamnet_model(params)
        if opt.weights and os.path.exists(opt.weights):
            yamnet.load_weights(opt.weights)
            for layer in yamnet.layers:
                layer.trainable = False
            o = Dense(units=n_classes, use_bias=True)(yamnet.layers[-3].output)
            o = Activation('softmax')(o) 
            model = Model(inputs=yamnet.input, outputs=o)
        else:
            model = yamnet

    elif opt.model_name == 'mobilenet_v3_tf':
        if opt.weights and os.path.exists(opt.weights):
            print(opt.weights)
            mobilenet_v3 = tf.keras.models.load_model(opt.weights)
            for layer in mobilenet_v3.layers:
                layer.trainable = False
            o = Dense(units=n_classes, use_bias=True)(mobilenet_v3.layers[-2].output)
            o = Activation('softmax')(o) 
            model = Model(inputs=mobilenet_v3.input, outputs=o)
        else:
            model = MobileNetV3Small(input_shape=(params.patch_frames, params.mel_bands), weights = None, classes=n_classes)
    elif opt.model_name == 'mobilenet_v3':
        model = MobileNetV3Small(input_shape=(params.patch_frames, params.mel_bands), weights = None, classes=n_classes)
    else
    i   return None

    return model


def build_dataset(opt):
    labels = None
    n_classes = 527 # default audioset classes
    print(opt.label_json)
    if opt.label_json and os.path.exists(opt.label_json):
        with open(opt.label_json) as f:
            json_data = f.read()
            labels = json.loads(json_data)
            n_classes = len(labels)

    train_generator = DataGenerator(opt.train_cache_dir, opt.batch_size, n_classes = n_classes)
    valid_generator = DataGenerator(opt.valid_cache_dir, opt.batch_size, n_classes = n_classes)

    return train_generator, valid_generator, labels, n_classes


def train(opt, model, train_generator, validation_data = None):
    model.summary()

    # Define training callbacks
    checkpoint = ModelCheckpoint(opt.model_out + opt.model_name + '.h5',
                    monitor='val_loss', 
                    verbose=1,
                    save_best_only=True, 
                    mode='auto')

    reducelr = ReduceLROnPlateau(monitor='val_loss', 
                factor=0.5, 
                patience=3, 
                verbose=1)

    earlystop = EarlyStopping(monitor='val_accuracy', patience = 10, verbose = 0, mode = 'auto')

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=0.001) #0.001 # SGD

    #categorical_crossentropy binary_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #categorical_accuracy 

    if opt.pre_train :
        model_history = model.fit(train_generator,
                            steps_per_epoch = len(train_generator),
                            batch_size = opt.batch_size,
                            epochs = opt.epochs,
                            #validation_data = valid_generator,
                            #validation_steps = len(valid_generator), #validation_generator
                            verbose = 1,
                            callbacks=[checkpoint])
    else:
        model_history = model.fit(train_generator,
                            steps_per_epoch = len(train_generator),
                            batch_size = opt.batch_size,
                            epochs = opt.epochs,
                            validation_data = valid_generator,
                            validation_steps = len(valid_generator), #validation_generator
                            verbose = 1,
                            callbacks=[earlystop,checkpoint,reducelr])

    model.save(opt.model_out + opt.model_name + '_last.h5')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=5, help='epochs defulat: 5')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size default: 1')
    parser.add_argument('--model_name', type=str, default='mobilenet_v3', help='use model name')
    parser.add_argument('--dataset_name', type=str, default='mine', help='use dataset name')
    parser.add_argument('--path', type=str, default='/home/ysr/dataset/audio/', help='dataset path')
    parser.add_argument('--model_out', type=str, default='saved_models/', help='dataset path')
    opt = parser.parse_args()

    opt.train_cache_dir = opt.path + opt.dataset_name + '/' + 'train.cache'
    opt.valid_cache_dir = opt.path + opt.dataset_name + '/' + 'valid.cache'
    opt.pre_train = True
    opt.label_json = None

    if opt.dataset_name == 'mine':
        opt.label_json = opt.path + opt.dataset_name + '/' + 'classes.json'
        opt.pre_train = False

    print(opt.train_cache_dir)
    print(opt.valid_cache_dir)
    print(opt.label_json)
    
    train_generator, valid_generator, labels, n_classes = build_dataset(opt)

    model = build_mode(opt, n_classes)
    train(opt, model, train_generator, valid_generator)
    
