import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adadelta

def ctc_loss(args):
    return K.ctc_batch_cost(*args)


labels_input = Input([None], dtype='int32')
sequential = Sequential([
    Reshape([32, -1, 1], input_shape=[32, None]),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    Permute((2, 1, 3)),
    TimeDistributed(Flatten()),
    LSTM(units=128, return_sequences=True),
    LSTM(units=128, return_sequences=True),
    TimeDistributed(Dense(10, activation='softmax'))
])
input_length = Lambda(lambda x: K.tile([[K.shape(x)[1]]], [K.shape(x)[0], 1]))(sequential.output)
label_length = Lambda(lambda x: K.tile([[K.shape(x)[1]]], [K.shape(x)[0], 1]))(labels_input)
output = Lambda(ctc_loss)([labels_input, sequential.output, input_length, label_length])
fit_model = Model(inputs=[sequential.input, labels_input], outputs=output)

adadelta = Adadelta(lr=0.05)
fit_model.compile(
    loss=lambda y_true, y_pred: y_pred,
    optimizer=adadelta)
fit_model.summary()
