import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _numpy_int32_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.reshape(-1)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(np.float32).tostring()]))


def _numpy_float32_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.reshape(-1)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(np.float32).tostring()]))

def _numpy_float64_feature(value):
    # Returns a byte_list from a string /byte 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.reshape(-1)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(np.float64).tostring()]))

def _float_feature(value):
    # Returns a float_list from a float / double 
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    # Returns a int64_list from a bool / enum / int /uint 
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    # Returns a int64_list from a bool / enum / int /uint 
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _shape_feature(value):
    # Returns a int64_list from a bool / enum / int /uint 
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

