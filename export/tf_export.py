import tensorflow.compat.v2 as tf

## pb 转 h5 tf2.0
def tf2keras(pb_path, h5_path):
    model = tf.keras.models.load_model(pb_path)
    tf.keras.models.save_model(model, h5_path, save_format='h5')
    model = tf.keras.models.load_model(h5_path)
    model.summary()

