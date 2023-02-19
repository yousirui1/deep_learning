import numpy as np
import tensorflow as tf
from tensorflow import keras
#from model.mobilenet_v3_small import MoblieNetV3_model, MobileNetV3_Small
from model.crnn import CRNN
import utils.params as yamnet_params
from tensorflow.keras.utils import to_categorical
from model.mobilenet_v3 import MobileNetV3
#from model.crnn_attention import CRNN_Attention
from sklearn.preprocessing import MultiLabelBinarizer

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("===============  GPU is ",tf.test.is_gpu_available(), "TF version", tf.__version__, "==================")

# 生成虚拟数据
x_train = np.random.random((1000, 96, 64))
y_train = np.random.randint(12, size=(1000, 1)) 
y_train = to_categorical(y_train)
x_test = np.random.random((100, 96, 64))
y_test = np.random.randint(12, size=(100, 1)) 
y_test = to_categorical(y_test)  # one hot 

# tf.v1 tensor -> numpy
#data_tensor = tf.convert_to_tensor(data_numpy)

#with tf.Session() as sess:
#    data_numpy = data_tensor.eval()

# tf.v2 tensor -> numpy
#data_tensor= tf.convert_to_tensor(data_numpy)
# eagertensor 
#tf.compat.v1.enable_eager_execution()
#data_numpy = data_tensor.numpy()

# numpy -> tensor array
#tf.constant(data_numpy)

#  multi label one hot
#classes = ['a', 'b']
#mlb = MultiLabelBinarizer(classes = classes)
#train_labels = mlb.fit_transform(train_labels)

def mobilenet_v3_test():
    params = yamnet_params.Params(num_classes = 12)
    mobilenet = MobileNetV3((params.patch_frames, params.patch_bands), params.num_classes)
    mobilenet.summary()
    mobilenet.callbacks(filename = "./out/mobilenet.h5", patience = 1, monitor = "val_accuracy")
    mobilenet.compile_(opt = 'adam')
    mobilenet.train(x_train, y_train, x_test, y_test)


def crnn_test():
    n_classes = 12 
    params = yamnet_params.Params(num_classes = 12)

    #model = CRNN_Attention((params.patch_frames, params.patch_bands), params.num_classes)
    model = CRNN(n_classes).build(params)
    model.summary()

    #model.callbacks(filename = "./out/mobilenet.h5", patience = 1, monitor = "val_accuracy")
    #model.compile_(opt = 'adam')
    #model.train(x_train, y_train, x_test, y_test)

    loss = 'categorical_crossentropy';    
    #optimizer = keras.optimizers.SGD(lr=self.opt.lr, decay=self.opt.weight_decay, momentum=self.opt.momentum, nesterov=True)
    #optimizer = keras.optimizers.SGD(lr=self.opt.lr, decay=self.opt.weight_decay, momentum=self.opt.momentum, nesterov=True)
    optimizer = keras.optimizers.Adam()

    model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
    model.fit(x_train, y_train,
                epochs = 30)

if __name__ == "__main__":
    #mobilenet_v3_test()
    crnn_test()
   # crnn_test()




