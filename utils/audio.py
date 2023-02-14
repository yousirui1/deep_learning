import soundfile as sf


def read_wav(wav_path, start_time = None, end_time = None, is_play = None):
    wav_data, sr = sf.read('/home/ysr/test/test.wav', dtype=np.int32)
    # 多通道改单通道
    wav_data= np.mean(wav_data, axis = 1, dtype=wav_data.dtype)

    if start_time != None and end_time != None:
        wav_data = wav_data[sr * start_time: sr *end_time]
    elif start_time == None and end_time != None:
        wav_data = wav_data[:sr * end_time]
    elif start_time != None and end_time == None:
        wav_data = wav_data[sr * start_time:]

    if is_play not None:
        display.Audio(wav_data)

    


