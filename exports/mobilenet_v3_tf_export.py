import os
import sys
sys.path.append('/home/ysr/project/ai/deep_learning/')
from utils.params import Params
import feature.audio_features as features_lib
import tensorflow as tf
import time

def log(msg):
  print('\n=====\n{} | {}\n=====\n'.format(time.asctime(), msg), flush=True)


def frames_model(params, model):
    """Defines the YAMNet waveform-to-class-scores model.

    Args:
        params: An instance of Params containing hyperparameters.

    Returns:
        A model accepting (num_samples,) waveform input and emitting:
        - predictions: (num_patches, num_classes) matrix of class scores per time frame
        - embeddings: (num_patches, embedding size) matrix of embeddings per time frame
        - log_mel_spectrogram: (num_spectrogram_frames, num_mel_bins) spectrogram feature matrix
    """
    waveform = tf.keras.layers.Input(batch_shape=(None,), dtype=tf.float32)
    waveform_padded = features_lib.pad_waveform(waveform, params)
    log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
              waveform_padded, params)
    net = tf.keras.layers.Reshape(
          (params.patch_frames, params.patch_bands, 1),
          input_shape=(params.patch_frames, params.patch_bands))(features)
    
    embeddings = model.layers[1](net)
    o = model.layers[2](embeddings)
    o = model.layers[3](o)
    o = model.layers[4](o)
    predictions = model.layers[5](o)
   
    # (type_spec=TensorSpec(shape=(None, 96, 64, 1)
    #predictions = model.input(net)
    #logits = tf.keras.layers.Dense(units=params.num_classes, use_bias=True)(embeddings)
    #predictions = tf.keras.layers.Activation(activation=params.classifier_activation)(logits)
    model = tf.keras.models.Model(
        name='frames_model', inputs=waveform,
        outputs=[predictions, embeddings, log_mel_spectrogram])
    model.summary()
    return model
  

class MobileNetv3(tf.Module):
    def __init__(self, embeddings, params):
        self._model = frames_model(params, embeddings)
        self._class_map_asset = tf.saved_model.Asset('export/yamnet_class_map.csv')
        self.params = params
    @tf.function(input_signature=[])
    def class_map_path(self):
        return self._class_map_asset.asset_path
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def __call__(self, waveform):
        predictions, embeddings, log_mel_spectrogram = self._model(waveform)
        
        return {'predictions': predictions,  
            'embeddings': embeddings, 
            'log_mel_spectrogram': log_mel_spectrogram}

def make_tflite_export(weights_path, export_dir, n_class):
    if os.path.exists(export_dir):
        log('TF-Lite export already exists in {}, skipping TF-Lite export'.format(
        export_dir))
        return

    # Create a TF-Lite compatible Module wrapper around YAMNet.  sigmoid
    log('Building and checking TF-Lite Module ...')
    params = Params(tflite_compatible=True, num_classes = n_class, classifier_activation = 'softmax')
    #yamnet = YAMNet(weights_path, params)
    model = tf.keras.models.load_model(weights_path)
    #embeddings = model.layers[1]
    yamnet = MobileNetv3(model, params)
    #check_model(yamnet, yamnet.class_map_path(), params)
    log('Done')

    # Make TF-Lite SavedModel export.
    log('Making TF-Lite SavedModel export ...')
    saved_model_dir = os.path.join(export_dir, 'saved_model')
    os.makedirs(saved_model_dir)
    tf.saved_model.save(
        yamnet, saved_model_dir,
          signatures={'serving_default': yamnet.__call__.get_concrete_function()})
    log('Done')

    # Check that the export can be loaded and works.
    log('Checking TF-Lite SavedModel export in TF2 ...')
    model = tf.saved_model.load(saved_model_dir)
    #check_model(model, model.class_map_path(), params)
    log('Done')

    # Make a TF-Lite model from the SavedModel.
    log('Making TF-Lite model ...')
    tflite_converter = tf.lite.TFLiteConverter.from_saved_model(
          saved_model_dir, signature_keys=['serving_default'])
    tflite_model = tflite_converter.convert()
    tflite_model_path = os.path.join(export_dir, 'yamnet.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    log('Done')

  # Check the TF-Lite export.
    log('Checking TF-Lite model ...')
    #interpreter = tf.lite.Interpreter(tflite_model_path)
    #runner = interpreter.get_signature_runner('serving_default')
    #check_model(runner, 'yamnet_class_map.csv', params)
    log('Done')

    return saved_model_dir


#os.remove()

output_dir = '/home/ysr/project/ai/deep_learning/saved_models'
tflite_export_dir = os.path.join(output_dir, 'tflite')
make_tflite_export('/home/ysr/project/ai/deep_learning/saved_models/mobilenet_v3_tf.h5', tflite_export_dir, 13)

#frames_model(params, embeddings)

#params = yamnet_params.Params(tflite_compatible=True, num_classes = 12, classifier_activation = 'softmax')
#model = tf.keras.models.load_model('/home/ysr/project/ai/deep_learning/saved_models/mobilenet_v3.h5')
#model.summary()
#embeddings = model.layers[1]
#model = MobileNetv3(embeddings, params)
#saved_model_dir = './saved_model_dir'
#tf.saved_model.save(
#      model, saved_model_dir,
#      signatures={'serving_default': model.__call__.get_concrete_function()})
 

#model = tf.keras.models.Sequential([
#        model.layers[1],
       # model.layers[1]
       # tf.keras.layers.Dense(12),
        #tf.keras.Activation('softmax'),
#])
#model.summary()

#params = yamnet_params.Params()




#model = tf.keras.models.load_model('/home/ysr/project/ai/deep_learning/saved_models/mobilenet_v3.h5')
#model.summary()
#embeddings = model.layers[1]


