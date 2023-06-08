import argparse
import os,io
from pathlib import Path
import sys
import ffmpeg
import subprocess as sp
import soundfile as sf
import numpy as np


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT / 'weights'

#if str(ROOT) not in sys.path
#   sys.path.append(str(ROOT))
#if str(ROOT / 'yolov7') not in sys.path
#   sys.path.append(str(ROOT / 'yolov7'))
#if str(ROOT / 'strong_sort') not in sys.path
#   sys.path.append(str(ROOT / 'stronge_sort'))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'
AUD_FORMATS = 'wav', 'mp3', 'ops', 'flac', 'aac'

def ffmpeg_audio(file_name):
    #probe = ffmpeg.probe(file_name)

    #duration_seconds = float(probe['format'][].encode('utf-8'))
    #interval = 10 # 10s
    #if(duration_length > 10):
    #    for i in range(duration_length):

    # ffmpeg get url pcm 

    FFMPEG_BIN = "ffmpeg"
    command_in = [ "ffmpeg",
                        '-i', file_name,
                        '-f', 's16le', #soundfile只支持wav格式
                        '-ar', '16000',
                        '-ac','1',
                        '-']

    pipe_in = sp.Popen(command_in, stdout = sp.PIPE, stderr=sp.PIPE)
    
    #FFMPEG_BIN = "ffmpeg"
    #command_in = [ 'ffmpeg',
    #            '-i', file_name,
    #            '-f', 'wav',
    #            '-ar', 16000,
    #            '-ac', '1',
    #            '-']
    print(command_in)
    #pipe_in = sp.Popen(command_in, stdout = sp.PIPE, stderr=sp.PIPE)

    while True:
        raw_audio = pipe_in.stdout.read(16000)  #从内存中读取音频流
        print("pipe_in read ok ");
        if (raw_audio == ''):
            break
        tmp_stream = io.BytesIO(raw_audio)  
        dat = np.frombuffer(tmp_stream.getbuffer()) 

        print(dat.shape)
        
        #print('from BytesIO', dat)

        #data, samplerate = sf.read(tmp_stream)  #soundfile只支持wav格式
        #raw_audio = pipe_in.stdout.read() #从内存中读取音频流
        #if(raw_audio == ''):
        #    break
        #tmp_stream = io.BytesIO(raw_audio)
        #data, samplerate = sf.read(tmp_stream)  # 

        #if(len(data) < 1):
        #    break
        print("ffmpeg recv data len")
        pipe_in.stdout.flush()
    pipe_in.stdout.close() 


def detect(source='0'):
    file_type = 0
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_file = Path(source).suffix[1:] in (AUD_FORMATS)

    print(is_file)

    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://', 'webrtc://'))
    print(is_url)

    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    if is_url and is_file:
        print('...download file')
        #source = check_file(source)
        
    ffmpeg_audio(source)

def onnx_inference(wav_file):
    import onnx
    import onnxruntime
    import numpy as np
    onnx_model_path = 'efficinet.onnx'
    onnx_model = onnx.load(onnx_model_path)

    onnxruntime_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())

    # 随机生成输入数据
    #input_shape = (1, 3000, 128)
    #input_data = np.random.rand(*input_shape).astype(np.float32)

    
    input_data = _wav2fbank(wav_file)    

    input_data = input_data.view(1, 998, 126).numpy()

    # 进行推理
    outputs = onnxruntime_session.run(None, {'inputs': input_data})

    # 处理输出结果
    output_results = outputs[0][0]  # 获取输出结果，假设结果形状为(1, 200)
     # 排序并输出前5个类别
    sorted_indices = np.argsort(output_results)[::-1]  # 按值排序并取逆序
    top5_indices = sorted_indices[:5]  # 取前5个类别的索引 

    print("Top 5 classes:")
    for index in top5_indices:
        print("Class:", index, "Probability:", output_results[index])

    return output_results

    # Run Detect
    #for 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)' )
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam' )
    opt = parser.parse_args()
    print(opt)

    detect(**vars(opt)) #**vars(opt) 返回对象object的属性和属性值的字典对象

