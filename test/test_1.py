import tensorflow as tf
import numpy as np
import onnx
import tf2onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# 创建一个简单的神经网络模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(2)
])

# 随机生成训练数据和标签
def generate_random_data():
    inputs = np.random.randn(100, 10)
    labels = np.random.randint(0, 2, (100,))
    return inputs, labels

# 编译和训练模型
model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
inputs, labels = generate_random_data()
model.fit(inputs, labels, epochs=10)

# 保存模型为HDF5格式
tf.keras.models.save_model(model, 'model.h5')

# 将模型转换为ONNX格式
onnx_model, _ = tf2onnx.convert.from_keras(model)  # 解包元组
onnx.save_model(onnx_model, 'model.onnx')

print("ONNX模型已导出：model.onnx")

# 将模型转换为PB格式
tf.saved_model.save(model, 'model.pb')
print("PB模型已导出：model.pb")

