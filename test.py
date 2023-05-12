import numpy as np
import tensorflow as tf
from tensorflow import keras
from model.crnn import CRNN
from utils.params import Params
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
from model.efficientnet import EfficientNetB0
from model.transformer_encoder import TransformerEncoder
from model.mobilenet_v3 import MobileNetV3Small
from exports.tflite_export import converer_keras_to_tflite_v2
from exports.tf_export import keras2tf
from exports.onnx_export import keras2onnx_export
from audioset.yamnet.yamnet import yamnet_model
#from tensorflow.keras.applications import EfficientNetB0
import onnx
import tf2onnx

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def log(msg):
  print('\n=====\n{} | {}\n=====\n'.format(time.asctime(), msg), flush=True)

print("===============  GPU is ",tf.test.is_gpu_available(), "TF version", tf.__version__, "==================")

# 生成虚拟数据
x_train = np.random.random((1000, 96, 64))
y_train = np.random.randint(12, size=(1000, 1)) 
y_train = to_categorical(y_train)
x_test = np.random.random((100, 96, 64))
y_test = np.random.randint(12, size=(100, 1)) 
y_test = to_categorical(y_test)  # one hot 

# tf.v1 numpy -> tensor
def tf_v1_numpy_to_tensor(data_numpy):
    data_tensor = tf.convert_to_tensor(data_numpy)

def tf_v1_tensor_to_numpy(data_tensor):
    with tf.Session() as sess:
        data_numpy = data_tensor.eval()

def tf_v2_numpy_to_tensor(data_numpy):
    data_tensor= tf.convert_to_tensor(data_numpy)

# tf.v2 tensor -> numpy
def tf_v2_tensor_to_numpy(data_tensor):
    # eagertensor 
    tf.compat.v1.enable_eager_execution()
    data_numpy = data_tensor.numpy()

# numpy -> tensor array
#tf.constant(data_numpy)

#  multi label one hot
#classes = ['a', 'b']
#mlb = MultiLabelBinarizer(classes = classes)
#train_labels = mlb.fit_transform(train_labels)

def mobilenet_v3_model(params):
    model = MobileNetV3Small(input_shape=(params.patch_frames, params.mel_bands), weights = None, classes=params.num_classes)
    return model

def efficientnet_model(params):
    inputs = keras.Input(shape=[params.patch_frames, params.mel_bands])
    x = keras.layers.Reshape((params.patch_frames, params.mel_bands, 1),
                    input_shape=(params.patch_frames, params.mel_bands))(inputs)

    efficientnet = EfficientNetB0(include_top=False, input_tensor=x, 
                    input_shape=(params.patch_frames, params.mel_bands, 1), weights = None)

    x = keras.layers.Flatten()(efficientnet.layers[-4].output)
    x = keras.layers.Dense(params.num_classes)(x)
    outputs = keras.layers.Activation('sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs = outputs, name="EfficientNet")
    return model

def transformer_encoder_model(params):
    # Create a simple model containing the encoder.
    inputs = keras.Input(shape=[params.patch_frames, params.mel_bands])
    #o = Reshape((params.patch_frames, params.patch_bands, 1),
    #            input_shape=(params.patch_frames, params.patch_bands))(inputs)
    o = keras.layers.LayerNormalization(axis=1)(inputs)
    o = keras.layers.Dropout(0.1)(o)

    embedding  = TransformerEncoder(
                intermediate_dim=768, num_heads= params.num_classes)(o)
    o = keras.layers.Dropout(0.1)(embedding)
    o = keras.layers.Flatten()(o)
    o = keras.layers.Dense(params.num_classes)(o)
    outputs = keras.layers.Activation('sigmoid')(o) 
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def crnn_model():

    return None


def acdnet_model(params):
    return None


def mine_model(params):
    inputs = keras.Input(shape=[params.patch_frames, params.mel_bands])
    x = keras.layers.Reshape((params.patch_frames, params.mel_bands, 1),
                    input_shape=(params.patch_frames, params.mel_bands))(inputs)
    x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.5)(x)
    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    outputs = keras.layers.Dense(units=params.num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model



    
if __name__ == "__main__":
    params = Params()
    #model = mobilenet_v3_model(params)
    #model = efficientnet_model(params)

    #model = mine_model(params)
    #model.summary()
    #model = transformer_encoder_model(params)
    model = yamnet_model(params)

    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
    model.fit(x_train, y_train,
                epochs = 2)

    model.save("saved_models/test.h5")
    onnx_model, _ = tf2onnx.convert.from_keras(model)
    onnx.save_model(onnx_model, 'saved_models/model.onnx')
    #keras2onnx_export("saved_models/test.h5", "saved_model/test.onnx")
    #keras.models.save_model(model, "saved_models/test", save_format='pb')
    #keras2tf("saved_models/test.h5", "saved_models/pb")
    #converer_keras_to_tflite_v2("saved_models/test.h5")

