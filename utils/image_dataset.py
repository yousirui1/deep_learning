cat_in_snow = '/home/ysr/mnt/image/test/000000000139.jpg'
tfrecord_file = '/home/ysr/mnt/image/train.tfrecords'
# This is an example, just using the cat image.
image_string = open(cat_in_snow, 'rb').read()

label = image_labels['cat_in_snow']

# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape    
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
          }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        
image_example(image_string, label)


# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)


raw_image_dataset = tf.data.TFRecordDataset(tfrecord_file)
image_dataset = raw_image_dataset.map(_parse_image_function)

import IPython.display as display

for image_features in image_dataset:
    image_raw = image_features['image_raw'].numpy()
    image_width = image_features['width']
    image_height = image_features['height']
    image_label = image_features['label']
    image_depth = image_features['depth']
    display.display(display.Image(data=image_raw))
    print(image_width)
    print(image_height)
    print(image_label)
    print(image_depth)



data_dir = '/home/ysr/mnt/image/test/'
tfrecord_file = data_dir + '../' + 'train.tfrecords'
train_filenames = [data_dir + filename for filename in os.listdir(data_dir)]

# Create a dictionary with features that may be relevant.
label = 0

def image_tfrecord(train_filenames,label):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for filename in train_filenames: 
            image_string = open(filename, 'rb').read()
            image_shape = tf.image.decode_jpeg(image_string).shape    
            feature = {
                'height': _int64_feature(image_shape[0]),
                'width': _int64_feature(image_shape[1]),
                'depth': _int64_feature(image_shape[2]),
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(image_string),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

image_tfrecord(train_filenames, label)            

raw_image_dataset = tf.data.TFRecordDataset(tfrecord_file)
image_dataset = raw_image_dataset.map(_parse_image_function)            

for image_features in image_dataset:
    image_raw = image_features['image_raw'].numpy()
    image_width = image_features['width']
    image_height = image_features['height']
    image_label = image_features['label']
    image_depth = image_features['depth']
    display.display(display.Image(data=image_raw))
    print(image_width)
    print(image_height)
    print(image_label)
    print(image_depth)

