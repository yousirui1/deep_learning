import os
import json
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Dropout, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from model.mobilenet_v3 import MobileNetV3Small
#from model.mobilenet_v3_small import MobileNetV3_Small, MoblieNetV3_model
from model.transformer_encoder import TransformerEncoder
from utils.datasets import DataGenerator
from utils.params import Params
from utils.tools import get_files_and_labels
from audioset.yamnet.yamnet import yamnet_model

print("TF GPU: {}".format(tf.test.gpu_device_name()), " enable: {}".format(tf.test.is_gpu_available()))
print("TF version:{}".format(tf.__version__))

def build_mode(opt):
    params = Params()
    n_classes = opt.n_classes
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
            mobilenet_v3 = tf.keras.models.load_model(opt.weights)
            #mobilenet_v3.summary()
            for layer in mobilenet_v3.layers:
                layer.trainable = False
            o = Dense(units=n_classes, use_bias=True)(mobilenet_v3.layers[-1].output)
            o = Activation('softmax')(o) 
            model = Model(inputs=mobilenet_v3.input, outputs=o)
            #mobilenet_v3.summary()
        else:
            model = MobileNetV3Small(input_shape=(params.patch_frames, params.mel_bands), weights = None, classes=n_classes)
    elif opt.model_name == 'mobilenet_v3':
        if opt.weights and os.path.exists(opt.weights):
            mobilenet_v3 = MobileNetV3Small(input_shape=(params.patch_frames, params.mel_bands), weights = None, classes=527)
            #mobilenet_v3.summary()        
            mobilenet_v3.load_weights(opt.weights)
            for layer in mobilenet_v3.layers:
                layer.trainable = False
            o = Conv2D(n_classes, kernel_size=1, padding='same', name='Logits')(mobilenet_v3.layers[-4].output)
            o = Flatten()(o)
            o = Activation('softmax')(o) 
            model = Model(inputs=mobilenet_v3.input, outputs=o)
        else:
            model = MobileNetV3Small(input_shape=(params.patch_frames, params.mel_bands), weights = None, classes=n_classes)
    elif opt.model_name == 'transformer_encoder':
        encoder = TransformerEncoder(
                    intermediate_dim=768, num_heads=12)

        # Create a simple model containing the encoder.
        inputs = keras.Input(shape=[params.patch_frames, params.mel_bands])
        #o = Reshape((params.patch_frames, params.patch_bands, 1),
        #            input_shape=(params.patch_frames, params.patch_bands))(inputs)
        o = keras.layers.LayerNormalization(axis=1)(inputs)
        o = keras.layers.Dropout(0.1)(o)
        embedding = encoder(o)
        o = keras.layers.Dropout(0.1)(embedding)
        o = keras.layers.Flatten()(o)
        o = keras.layers.Dense(n_classes)(o)
        outputs = keras.layers.Activation('sigmoid')(o) 
        model = keras.Model(inputs=inputs, outputs=outputs)
    else:
       return None

    return model


def build_dataset(opt):
    files_train, files_val, labels = get_files_and_labels(opt.path + "train_npy/", file_type = 'npy', train_split=1.0,
                                single_cls= opt.single_cls)

    train_generator = DataGenerator(files_train,
                                labels,
                                batch_size = opt.batch_size,
                                n_classes = len(labels))

    valid_generator = DataGenerator(files_val,
                                    labels,
                                    batch_size= opt.batch_size,
                                    n_classes = len(labels))

    n_classes = len(labels)
    return train_generator, valid_generator, labels, n_classes


def train(opt, model, train_generator, validation_data = None):
    model.summary()

    # Define training callbacks
    checkpoint = ModelCheckpoint(opt.model_out + opt.model_name + '.h5',
                    monitor='loss', 
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
    #optimizer = tf.keras.optimizers.SGD() #0.001 # SGD

    #categorical_crossentropy binary_crossentropy

    if opt.pre_train :
        print(' --------------  pre train loss=binary_crossentropy ---------------')
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #categorical_accuracy 
        model_history = model.fit(train_generator,
                            steps_per_epoch = len(train_generator),
                            batch_size = opt.batch_size,
                            epochs = opt.epochs,
                            #validation_data = valid_generator,
                            #validation_steps = len(valid_generator), #validation_generator
                            verbose = 1,
                            callbacks=[checkpoint])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #categorical_accuracy 
        model_history = model.fit(train_generator,
                            steps_per_epoch = len(train_generator),
                            batch_size = opt.batch_size,
                            epochs = opt.epochs,
                            validation_data = valid_generator,
                            validation_steps = len(valid_generator), #validation_generator
                            verbose = 1,
                            #callbacks=[earlystop,checkpoint,reducelr])
                            callbacks=[checkpoint, reducelr])

    model.save(opt.model_out + opt.model_name + '_last.h5')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=5, help='epochs defulat: 5')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size default: 1')
    parser.add_argument('--model_name', type=str, default='mobilenet_v3', help='use model name')
    parser.add_argument('--dataset_name', type=str, default='esc-50', help='use dataset name')
    parser.add_argument('--path', type=str, default='', help='dataset path')
    parser.add_argument('--model_out', type=str, default='saved_models/', help='dataset path')
    parser.add_argument('--single-cls', action='store_false', help='train multi-class data as single-class')
    opt = parser.parse_args()

    #opt.train_cache_dir = opt.path + opt.dataset_name + '/' + 'train.cache'
    #opt.valid_cache_dir = opt.path + opt.dataset_name + '/' + 'valid.cache'

    opt.pre_train = True
    opt.label_json = None
    opt.wanted_label = ''

    opt.path = '/home/ysr/dataset/audio/' + opt.dataset_name + '/'


    #if opt.dataset_name == 'mine':
    #    opt.label_json = opt.path + opt.dataset_name + '/' + 'classes.json'
    #    opt.pre_train = False

    #opt.path = '/home/ysr/project/ai/yamnet-transfer-learning/train_set_patches/'
    #print(opt.train_cache_dir)
    #print(opt.valid_cache_dir)
    #print(opt.label_json)
    
    #train_generator, valid_generator, labels, n_classes = build_dataset_np(opt)
    train_generator, valid_generator, labels, n_classes = build_dataset(opt)
    opt.n_classes = n_classes

    model = build_mode(opt)
    model.summary()
    train(opt, model, train_generator, valid_generator)
    
