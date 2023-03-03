
def log(msg):
    print('\n=====\n{} | {}\n=====\n'.format(time.asctime(), msg), flush=True)


class mobilenet_v3_model(param):
    """Defines the YAMNet waveform-to-class-scores model.

    Args:
        params: An instance of Params containing hyperparameters.

    Returns:
        A model accepting (num_samples,) waveform input and emitting:
        - predictions: (num_patches, num_classes) matrix of class scores per time frame
        - embeddings: (num_patches, embedding size) matrix of embeddings per time frame
        - log_mel_spectrogram: (num_spectrogram_frames, num_mel_bins) spectrogram feature matrix
    """
    waveform = layers.Input(batch_shape=(None,), dtype=tf.float32)
    waveform_padded = features_lib.pad_waveform(waveform, params)
    log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
                    waveform_padded, params)

    embeddings = load();

    input->embedding-





class MobileNetV3Small(tf.Module):
    """ A TF2 Module wrapper around MobileNetv3 """
    def __init__(self, weights_path, params):

    

def make_tflite_export(weights_path, export_dir, n_class):
    if os.path.exists(export_dir):
        log('TF-Lite export already exists in {}, skipping TF-Lite export'.format(
            export_dir))
        return

    log('Building and checking TF-Lite Module ...')
    model = ();

    # Make TF-Lite SavedModel export
    log('Making TF-Lite SavedModel export ...')
    saved_model_dir = os.path.join(export_dir, 'saved_model')
    tf.saved_model.save(
            model, saved_model_dir,
            signatures={'serving_default': yamnet.__call__.get_concrete_function()})
    log('Done')

