"""MobileNet v3 small models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape,Dense,Activation
from tensorflow.keras.models import Model
#from tensorflow.keras.utils.vis_utils import plot_model

#from keras.models import Model
#from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape
#from keras.utils.vis_utils import plot_model

from model.mobilenet_base import MobileNetBase
import feature.audio_features as features_lib


def MoblieNetV3_model(params):
    shape = (96, 64)
    n_class = int(10)
    waveform = Input(batch_shape=(None,), dtype=tf.float32)
    waveform_padded = features_lib.pad_waveform(waveform, params)
    log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
      waveform_padded, params)
    
    #x = MobileNetV3_Small(shape, n_class).build(params, features)
    #model = Model(name = 'moblienet_v3_samll', inputs = waveform,
    #                    outputs = [x])
    return model

class MobileNetV3_Small(MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
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
        super(MobileNetV3_Small, self).__init__(shape, n_class, alpha)
        self.include_top = include_top


    def build(self, params, features, plot=False):
        """build MobileNetV3 Small.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        #inputs = Input(shape=self.shape)


        # tensorflow ??? keras ????????????
    

        #features = Input((params.patch_frames, params.patch_bands), dtype=tf.float32)
        net = Reshape((params.patch_frames, params.patch_bands, 1),
                 input_shape=(params.patch_frames, params.patch_bands))(features)

        x = self._conv_block(net, 16, (3, 3), strides=(2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')

        x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        #x = self._conv_block(x, 1280, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 576))(x)

        #x = Conv2D(1280, (1, 1), padding='same')(x)
        x = Conv2D(1280, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')

        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
            x = Reshape((self.n_class,))(x)
        
        #model = Model(features, x)
        #x = Conv2D(self.n_class, (1, 1), padding='same')(x)
        #x = Dense(units=self.n_class, use_bias=True)(x)
        #x = Activation('sigmoid')(x) #sigmoid
        #x = Activation('sigmoid')(x) #sigmoid

        #if plot:
       #     plot_model(model, to_file='images/MobileNetv3_small.png', show_shapes=True)

        return x

    def model(self,params):
        inputs = layers.Input(shape=(params.patch_frames, params.patch_bands))
        predictions = self.build(params, inputs)
        model = Model(name='moblienet_v3', inputs=inputs, outputs=predictions)
        return model
    
    def frames_model(self, params):
        waveform = Input(batch_shape=(None,), dtype=tf.float32)
        waveform_padded = features_lib.pad_waveform(waveform, params)
        log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
          waveform_padded, params)
        predictions = self.build(params, features)
        frames_model = Model(
            name = 'moblienet_frames', inputs=waveform,
            outputs = [predictions, log_mel_spectrogram])
        
    #model = Model(name = 'moblienet_v3_samll', inputs = waveform,
    #                    outputs = [x])
        
        return frames_model
