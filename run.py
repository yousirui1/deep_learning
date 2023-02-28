import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from tensorflow.keras.models import Model
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
#from model.mobilenet_v3_small import MobileNetV3_Small, MoblieNetV3_model
from model.mobilenet_v3 import MobileNetV3Small
from utils.datasets import DataGenerator, get_files_and_labels, DataGenerator_NP
import utils.params as yamnet_params

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("TF GPU: {}".format(tf.test.gpu_device_name()), " enable: {}".format(tf.test.is_gpu_available()))
print("TF version:{}".format(tf.__version__))


def train_test(opt):
    train_dir = '/home/ysr/project/ai/yamnet-transfer-learning/train_set_patches/'
    model_out = './saved_models/model'
    model_last_out = './saved_models/last'
    batch_size = 1
    epochs = 100

    files_train, files_val, labels = get_files_and_labels(train_dir, 
                                                    typ='npy',
                                                    train_split=0.8,
                   # wanted_label = 'Alarm,ChainSaw,Cough,Cry,Explosion,GlassBreak,Knock,Scream,Siren,Voice' #Firecracker
                    #               ',Buzzer,Glass,Laughter,Acoustic,Carhorn,Music') #HandSaw Applause
                 wanted_label = 'Alarm,ChainSaw,Cough,GlassBreak,Buzzer,Explosion,Siren,Digging,Smash,Voice')            

    train_generator = DataGenerator_NP(files_train,
                                labels,
                                batch_size=batch_size,
                                n_classes = len(labels))

    valid_generator = DataGenerator_NP(files_val,
                                    labels,
                                    batch_size=batch_size,
                                    n_classes = len(labels))

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


    if opt.weights and os.path.exists(opt.weights):
        model = tf.keras.models.load_model('saved_models/mobilenet_v3')
        tf.keras.models.save_model(model, opt.weights, save_format='h5')
        model = tf.keras.models.load_model(opt.weights)
        model.summary()
        #model.load_weights(opt.weights)
        for layer in model.layers:
            layer.trainable = False
        #o = tf.keras.layers.Reshape((len(labels),))(o)
        #tf.keras.layers.Dense
        #o = tf.keras.layers.Reshape((len(labels)))(model.layers[-2].output)
        #o = tf.keras.layers.Flatten(12, (1, 1), padding='same', activation='softmax')(model.layers[-3].output)
        #o = tf.keras.layers.Reshape((12,))(o)
        #o = tf.keras.layers.Conv2D(len(labels), (1, 1), padding='same', activation='softmax')(model.layers[-4].output)
        o = tf.keras.layers.Conv2D(12,
                          kernel_size=1,
                          padding='same',
                          name='Logits')(model.layers[-4].output)
        o = tf.keras.layers.Flatten()(o)
        o = tf.keras.layers.Softmax(name='Predictions/Softmax')(o)

        model = Model(inputs=model.input, outputs=[o])
        model.summary()

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=0.001) #0.001
    #optimizer = tf.keras.optimizers.SGD() #0.001
    #categorical_crossentropy binary_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #categorical_accuracy 
    model_history = model.fit(train_generator,
                            steps_per_epoch = len(train_generator),
                            batch_size = batch_size,
                            epochs = epochs,
                            validation_data = valid_generator,
                            validation_steps = len(valid_generator), #validation_generator
                            verbose = 1,
                            callbacks=[checkpoint])
                            #callbacks=[earlystop,checkpoint,reducelr])
                            #callbacks=[checkpoint])

    model.save(model_last_out+'.h5')


    


def train_np(opt):
    train_dir = '/home/ysr/project/ai/yamnet-transfer-learning/train_set_patches/'
    model_out = './saved_models/model'
    model_last_out = './saved_models/last'
    batch_size = 1
    epochs = 100

    files_train, files_val, labels = get_files_and_labels(train_dir, 
                                                    typ='npy',
                                                    train_split=0.8,
                   # wanted_label = 'Alarm,ChainSaw,Cough,Cry,Explosion,GlassBreak,Knock,Scream,Siren,Voice' #Firecracker
                    #               ',Buzzer,Glass,Laughter,Acoustic,Carhorn,Music') #HandSaw Applause
                 wanted_label = 'Alarm,ChainSaw,Cough,GlassBreak,Buzzer,Explosion,Siren,Digging,Smash,Voice')            

    train_generator = DataGenerator_NP(files_train,
                                labels,
                                batch_size=batch_size,
                                n_classes = len(labels))

    valid_generator = DataGenerator_NP(files_val,
                                    labels,
                                    batch_size=batch_size,
                                    n_classes = len(labels))

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
    #model = MobileNetV3_Small((params.patch_frames, params.mel_bands), 527).model(params)
    model = MobileNetV3Small(input_shape=(params.patch_frames, params.mel_bands), weights = None, classes=527)
    model.summary()

    if opt.weights and os.path.exists(opt.weights):
        model.load_weights(opt.weights)
        for layer in model.layers:
            layer.trainable = False
        #o = tf.keras.layers.Reshape((len(labels),))(o)
        #tf.keras.layers.Dense
        #o = tf.keras.layers.Reshape((len(labels)))(model.layers[-2].output)
        #o = tf.keras.layers.Flatten(12, (1, 1), padding='same', activation='softmax')(model.layers[-3].output)
        #o = tf.keras.layers.Reshape((12,))(o)
        #o = tf.keras.layers.Conv2D(len(labels), (1, 1), padding='same', activation='softmax')(model.layers[-4].output)
        o = tf.keras.layers.Conv2D(12,
                          kernel_size=1,
                          padding='same',
                          name='Logits')(model.layers[-4].output)
        o = tf.keras.layers.Flatten()(o)
        o = tf.keras.layers.Softmax(name='Predictions/Softmax')(o)

        model = Model(inputs=model.input, outputs=[o])
        model.summary()


#yer in yamnet.layers:
    #print(layer)
#     layer.trainable = False

#o = tf.keras.layers.Dense(units=len(labels), use_bias=True)(yamnet.layers[-3].output)
#o = tf.keras.layers.Activation('softmax')(o) #sigmoid

#if tflite_ouput == 1:
#    model = Model(inputs=yamnet.input, outputs=o)
#else:
#    model = Model(inputs=yamnet.input, outputs=[o, yamnet.output])

#model.summary()


    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=0.001) #0.001
    #optimizer = tf.keras.optimizers.SGD() #0.001
    #categorical_crossentropy binary_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #categorical_accuracy 
    model_history = model.fit(train_generator,
                            steps_per_epoch = len(train_generator),
                            batch_size = batch_size,
                            epochs = epochs,
                            validation_data = valid_generator,
                            validation_steps = len(valid_generator), #validation_generator
                            verbose = 1,
                            callbacks=[checkpoint])
                            #callbacks=[earlystop,checkpoint,reducelr])
                            #callbacks=[checkpoint])

    model.save(model_last_out+'.h5')



def train(hyp, opt):
    train_cache_dir = '/home/ysr/mnt/audio/audio_set/train.cache'
    #valid_cache_dir = '/home/ysr/mnt/audio/dataset/audio/audioset/valid.cache'
    n_classes = 527

    model_out = './saved_models/model'
    model_last_out = './saved_models/last'
    batch_size = 1
    epochs = opt.epochs

    train_generator = DataGenerator(train_cache_dir, batch_size, train_split = 0.3, n_classes = n_classes)
    #valid_generator = DataGenerator(valid_cache_dir, batch_size, train_split = 0.1,  n_classes = n_classes)

    # Define training callbacks
    checkpoint = ModelCheckpoint(model_out+'.h5',
              monitor='loss', 
              verbose=1,
              save_best_only=True, 
              mode='auto')

    reducelr = ReduceLROnPlateau(monitor='val_loss', 
                factor=0.5, 
                patience=3, 
                verbose=1)

    earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=0, mode='auto')

    params = yamnet_params.Params()
    #model = MobileNetV3_Small((params.patch_frames, params.mel_bands), n_classes).model(params)
    model = MobileNetV3Small(input_shape=(params.patch_frames, params.mel_bands), weights = None, classes=n_classes)
    if opt.weights and os.path.exists(opt.weights):
        print(opt.weights)
        model.load_weights(opt.weights)

    model.summary()

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr = 0.001) #0.001
    #optimizer = tf.keras.optimizers.SGD(lr=0.001) #0.001
    #categorical_crossentropy binary_crossentropy
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #categorical_accuracy 
    #model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #categorical_accuracy 
    model_history = model.fit(train_generator,
                            steps_per_epoch = len(train_generator),
                            batch_size = batch_size,
                            epochs = epochs,
                            #validation_data = valid_generator,
                            #validation_steps = len(valid_generator), #validation_generator
                            verbose = 1,
                            callbacks=[checkpoint])
#                            callbacks=[earlystop,checkpoint,reducelr])
    model.save(model_last_out+'.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=5, help='epochs 10')
    opt = parser.parse_args()
    train(None, opt)
    #train_np(opt)
    #train_test(opt)

