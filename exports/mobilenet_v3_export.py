import os
import sys
import time

import numpy as np
import tensorflow as tf
import model.params as yamnet_params

from model.mobilenet_v3_small import MobileNetV3_Small

def log(msg):
    print('\n===========\n{} | {} \n ============\n'.format(time.asctime(),msg), flush=True)
    
assert tf.version.VERSION >= '2.0.0', (
    'Need at least TF 2.0, you have TF v{}'.format(tf.version.VERSION))


class MoblieNetV3(tf.Module):
    """ A TF2 Module wrapper around MoblieNetV3. """
    def __init__(self, weights_path, params, n_class):
        super().__init__()
        self._moblienet = MobileNetV3_Small((params.patch_frames, params.mel_bands), n_class).frames_model(params)
        self._moblienet.load_weights(weights_path)
        self._class_map_asset = tf.saved_model.Asset('yamnet_class_map.csv')
        self.params = params
        
    @tf.function(input_signature = [] )
    def class_map_path(self):
        return self._class_map_asset.asset_path
    
    @tf.function(input_signature = [tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def __call__(self, waveform):
        predictions, log_mel_spectrogram = self._moblienet(waveform)
        return {'predictions' : predictions,
               'log_mel_spectrogram' : log_mel_spectrogram}
        

def make_tflite_export(weights_path, export_dir, n_class):
    if os.path.exists(export_dir):
        log('TF-Lite export already exists in {}, skipping TF-Lite export'.format(export_dir))
        return
    
    # Create a TF-Lite compatible Module wrapper around MoblieNetV3
    log('Building and checking TF-Lite Module ...')
    params = yamnet_params.Params(tflite_compatible=True, num_classes = n_class, classifier_activation = 'softmax')
    moblienet_v3 = MoblieNetV3(weights_path, params, n_class) 
    log('Done')
    
    # Make TF-Lite SavedModel export
    log('Making TF-Lite SavedModel export ...')
    saved_model_dir = os.path.join(export_dir, 'saved_model')
    os.makedirs(saved_model_dir)
    tf.saved_model.save(
        moblienet_v3, saved_model_dir,
        signatures = {'serving_default': moblienet_v3.__call__.get_concrete_function()})
    log('Done')
    
    # Check that the export can be loaded and work
    log('Checking TF-Lite SavedModel export in TF2 ...')
    #model = tf.saved_mode.load()
    #check()
    log('Done')
    
    # Make a TF-Lite model from the SavedModel.
    log('Making TF-Lite model ...')
    tflite_converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_model_dir, signature_keys = ['serving_default'])
    tflite_model = tflite_converter.convert()
    tflite_model_path = os.path.join(export_dir, 'moblienet_v3.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    log('Done')
    
    # Check the TF-Lite export 
    log('Checking TF-Lite model ...')
    # 
    log('Done')
    
    return saved_model_dir
print(os.getcwd)
os.rmdir(os.getcwd + "/saved_models/tflite")
output_dir = './saved_models'
tflite_export_dir = os.path.join(output_dir, 'tflite')
tflite_saved_model_dir = make_tflite_export('./saved_models/model.h5', tflite_export_dir, 2)
