
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Dropout, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from model.mobilenet_v3 import MobileNetV3Small
from model.efficientnet import EfficientNetB0
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
            mobilenet_v3.summary()
            for layer in mobilenet_v3.layers:
                layer.trainable = False
            o = keras.layers.Dropout(0.1)(mobilenet_v3.layers[-2].output)
            #o = keras.layers.Dropout(0.1)(mobilenet_v3.layers[-4].output)
            o = Dense(units=n_classes, use_bias=True)(o)
            o = Activation('softmax')(o)
            model = Model(inputs=mobilenet_v3.input, outputs=o)
            #mobilenet_v3.summary()
        else:
            model = MobileNetV3Small(input_shape=(params.patch_frames, params.mel_bands), weights = None, classes=n_classes)
    elif opt.model_name == 'mobilenet_v3':
        if opt.weights and os.path.exists(opt.weights):
            mobilenet_v3 = MobileNetV3Small(include_top=False, input_shape=(params.patch_frames, params.mel_bands), weights = None, classes=10)
            mobilenet_v3.summary()
            mobilenet_v3.load_weights(opt.weights)
            #for layer in mobilenet_v3.layers:
            #    layer.trainable = False
            o = Conv2D(n_classes, kernel_size=1, padding='same', name='Logits')(mobilenet_v3.layers[-2].output)
            o = Flatten()(mobilenet_v3.layers[-2].output)
            o = Dense(units=n_classes, use_bias=True)(o)
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
    elif opt.model_name == 'efficientnet':
        inputs = keras.Input(shape=[params.patch_frames, params.mel_bands])

        o = Reshape((params.patch_frames, params.patch_bands, 1),
                    input_shape=(params.patch_frames, params.patch_bands))(inputs)

        efficientnet = EfficientNetB0(include_top=False, input_tensor=o, input_shape=(params.patch_frames, params.patch_bands, 1),
                                         weights=None)

        #efficientnet.load_weights(opt.weights)

        #efficientnet = tf.keras.models.load_model(opt.weights)

        #x = keras.layers.Dense(n_classes)(efficientnet.layers[-1].output)
        #outputs = keras.layers.Activation('sigmoid')(x) 

        #efficientnet.summary()

        #o = Dense(units=n_classes, use_bias=True)(mobilenet_v3.layers[-1].output)
        #o = Activation('softmax')(o) 
        #model = Model(inputs=mobilenet_v3.input, outputs=o)

        # Rebuild top
        #x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(efficientnet.layers[-1].output)
        #x = keras.layers.BatchNormalization()(x)

        #top_dropout_rate = 0.1
        #x = keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        x = keras.layers.Flatten()(efficientnet.layers[-4].output)
        x = keras.layers.Dense(n_classes)(x)
        outputs = keras.layers.Activation('sigmoid')(x)
        #outputs = keras.models.layers.Dense(n_classes, activation="softmax", name="pred")(x)
        # Compile
        #model = tf.keras.Model(efficientnet.input, ouput = x, name="EfficientNet")
        model = Model(inputs=efficientnet.input, outputs = outputs, name="EfficientNet")
    else:
       return None

    return model

def build_dataset(opt):
    if opt.single_cls == True:
        files_train, files_val, labels = get_files_and_labels(opt.path + "train_npy/", file_type = 'npy', train_split=0.9,
                                wanted_label = opt.wanted_label, single_cls= opt.single_cls)
        n_classes = len(labels)
    else:
        files_train, files_val, labels = get_files_and_labels(opt.path + "train_npy/patches/", file_type = 'npy',
                                train_split=0.8, wanted_label = opt.wanted_label, single_cls= opt.single_cls)
        n_classes = 527

    train_generator = DataGenerator(files_train,
                                labels,
                                batch_size = opt.batch_size,
                                n_classes = len(labels),
                                single_cls = opt.single_cls)

    valid_generator = DataGenerator(files_val,
                                    labels,
                                    batch_size= opt.batch_size,
                                    n_classes = len(labels),
                                    single_cls = opt.single_cls)

    print("len ", len(train_generator))
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
    optimizer = tf.keras.optimizers.Adam(lr=0.1) #0.001 # SGD
    #optimizer = tf.keras.optimizers.SGD() #0.001 # SGD

    #categorical_crossentropy binary_crossentropy

    if opt.pre_train :
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #categorical_accuracy 
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
                            #callbacks=[checkpoint])

    model.save(opt.model_out + opt.model_name + '_last.h5')
