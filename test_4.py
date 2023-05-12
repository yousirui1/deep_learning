import numpy as np
import onnx
import onnxruntime
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import tf2onnx

# 定义CRNN模型
def create_crnn_model(input_shape, num_classes):
    input_data = keras.Input(shape=input_shape)

    # CNN部分
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_data)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # 调整形状以适应RNN
    x = Reshape(target_shape=(input_shape[0] // 4, (input_shape[1] // 4) * 64))(x)

    # RNN部分
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Bidirectional(GRU(128, return_sequences=True))(x)

    # 输出层
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_data, outputs=x)
    return model

# 创建CRNN模型
input_shape = (32, 128, 3)
num_classes = 10
model = create_crnn_model(input_shape, num_classes)

# 导出模型为ONNX格式
#onnx_model = tf.keras2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
#onnx.save_model(onnx_model, 'crnn.onnx')
onnx_model, _ = tf2onnx.convert.from_keras(model)  # 解包元组
onnx.save_model(onnx_model, 'model.onnx')

