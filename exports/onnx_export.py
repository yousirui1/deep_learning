import tensorflow as tf
from tensorflow import keras
import keras2onnx
import onnx

def keras2onnx_export(h5_path, onnx_path):
    keras_model = keras.models.load_model(h5_path)
    keras_model.summary()
    onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
    onnx.save_model(onnx_model, onnx_path)


