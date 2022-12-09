import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, LSTM,Permute,TimeDistributed,MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation, GRU, Lambda, add, concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


class CRNN():
    def __init__(self, n_class, include_top=True):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.

        # Returns
            MobileNetv3 model.
        """
        #super(CRNN, self).__init__(shape, n_class)
        self.include_top = include_top
        self.n_class = n_class



    def build(self, params, plot=False):
        """build MobileNetV3 Small.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        features = Input((params.patch_frames, params.patch_bands), dtype=tf.float32)
        net = Reshape((params.patch_frames, params.patch_bands, 1),
                 input_shape=(params.patch_frames, params.patch_bands))(features)

        #encoder_embedded = Embedding(num_input_tokens, embedding_size, mask_zero=True)(net)

         # Convolution layer (VGG)
        inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(net)  # (None, 128, 64, 64)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

        inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

        inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

        inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

        inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)

        # CNN to RNN
        inner = Reshape((192, 256), name='reshape2')(inner)  # (None, 32, 2048)
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

        # RNN layer
        gru_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)  # (None, 32, 512)
        gru_1b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
        reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_1b)

        gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 32, 512)
        gru1_merged = BatchNormalization()(gru1_merged)

        gru_2 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
        reversed_gru_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_2b)

        gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 32, 1024)
        gru2_merged = BatchNormalization()(gru2_merged)

        # transforms RNN output to character activations:
        inner = Dense(self.n_class, name='dense2')(gru2_merged) #(None, 32, 63)
        inner = Flatten()(inner)
        inner = Dense(self.n_class, kernel_initializer='he_normal',name='dense3')(inner) #(None, 32, 63)

        y_pred = Activation('softmax', name='softmax')(inner)
        #y_pred = TimeDistributed(Dense(self.n_class, activation='softmax'))(gru2_merged)

        model = Model(features, y_pred)
        return model

