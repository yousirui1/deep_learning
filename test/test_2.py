import tensorflow as tf
import numpy as np
import tf2onnx
import onnx

# 随机生成训练数据和标签
def generate_random_data():
    inputs = np.random.randn(100, 224, 224, 3)
    labels = np.random.randint(0, 1000, (100,))
    return inputs, labels

# 创建MobileNet模型
model = tf.keras.applications.MobileNet(weights=None, classes=1000)

# 定义损失函数和优化器
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

# 随机初始化模型参数
model.summary()
model.compile(loss=loss_fn,
                             optimizer=optimizer,
                            metrics=['accuracy'])

x,y = generate_random_data()
model.fit(x, y, epochs=2)

# 将模型转换为ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save_model(onnx_model, 'mobilenet.onnx')

print("ONNX模型已导出：mobilenet.onnx")

