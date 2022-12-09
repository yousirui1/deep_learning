from rknn.api import RKNN
from tensorflow.python.ops import gen_audio_ops as audio_ops
import tensorflow as tf
import numpy as np

wav_file = open("recoard.wav", "rb")
wav_data = wav_file.read()
decoded_sample_data = audio_ops.decode_wav(wav_data, desired_channels=1, desired_samples=16000, name='decoded_sample_data')
spectrogram = audio_ops.audio_spectrogram(decoded_sample_data.audio, window_size=480, stride=160, magnitude_squared=True)
#fingerprint_input = audio_ops.mfcc(decoded_sample_data, 16000,  dct_coefficient_count=40) # 40 取40 个点
                                                                                # shape = (1,98, 40)
  
#with tf.Session() as sess:
#   fingerprint_input_npy = fingerprint_input.eval()                                                                              # 一维矩阵 40 个

print(fingerprint_input)
fingerprint_input_npy = fingerprint_input.numpy()
np.save('fingerprint_input.npy',fingerprint_input_npy)
np.savetxt('fingerprint_input.txt',fingerprint_input_npy)
print(fingerprint_input_npy)
# Create RKNN object
rknn = RKNN()
# Load TensorFlow Model
ret = rknn.load_rknn(path='./speech_command.rknn')


print("rknn runtime start")
ret = rknn.init_runtime(perf_debug=False)
outputs, = rknn.inference(inputs=fingerprint_input_npy,data_type='float32')
print("rknn runtime stop")
#outputs = rknn.inference(inputs=[fingerprint_input])
# Release RKNN Context
rknn.release()


#On-board microphone
#subprocess.call(["arecord", "-D", "MainMicCapture", "-r", "44100", "-c", "2", "-f", "S16_LE", "-d", "1", "record.wav"])
#subprocess.call(["sox", "record.wav", "-r", "16000","-c", "1", "record_16k.wav","remix","1"])
            
#external microphone
#subprocess.call(["arecord", "-D", "FreeMicCapture", "-r", "44100", "-c", "2", "-f", "S16_LE", "-d", "1", "record.wav"])
#subprocess.call(["sox", "record.wav", "-r", "16000","-c", "1", "record_16k.wav","remix","2"])


# 特征提取
#wav_file = open("record_16k.wav", "rb")
#wav_data = wav_file.read()
#decoded_sample_data = contrib_audio.decode_wav(wav_data, desired_channels=1, desired_samples=16000, name='decoded_sample_data')
#spectrogram = contrib_audio.audio_spectrogram(decoded_sample_data.audio, window_size=480, stride=160, magnitude_squared=True)
#fingerprint_input = contrib_audio.mfcc(spectrogram, 16000,  dct_coefficient_count=40)

## 推理
# Create RKNN object
#rknn = RKNN()
# Load TensorFlow Model
#ret = rknn.load_rknn(path='./speech_command.rknn')
#ret = rknn.init_runtime(perf_debug=False)
#outputs, = rknn.inference(inputs=fingerprint_input_npy,data_type='float32')
# Release RKNN Context
#rknn.release()

def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.io.gfile.GFile(filename)]

## 后处理
labels = load_labels("./conv_labels.txt")
predictions = np.array(outputs)
top_k = predictions[0].argsort()[-3:][::-1]
print(top_k)
for node_id in top_k:
    human_string = labels[node_id]
    score = predictions[0][node_id]
    print('%s (score = %.5f)' % (human_string, score))

