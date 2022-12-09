import contextlib
import wave
import webrtcvad
import os
import sys
import collections
MODE = 3
def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:s
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate
def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, vad, frames):

    voiced_frames = []
    for idx, frame in enumerate(frames):
        print(len(frames))
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        print(is_speech)
        if is_speech:
            voiced_frames.append(frame)
    return b''.join(f.bytes for f in voiced_frames)

def voiced_frames_expand(voiced_frames, duration=2):
    total = duration * 16000
    expand_voiced_frames = voiced_frames
    while len(expand_voiced_frames) < total:
        expand_num = total - len(expand_voiced_frames)
        expand_voiced_frames += voiced_frames[: expand_num]
    return expand_voiced_frames

def filter(wavpath, out_dir, expand=False):
    audio, sample_rate = read_wave(wavpath)
    vad = webrtcvad.Vad(MODE)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    voiced_frames = vad_collector(sample_rate, vad, frames)
    voiced_frames = voiced_frames_expand(voiced_frames, 2)
    wav_name = wavpath.split('/')[-1]
    #save_path = out_dir + '\\'  +"mo"+wav_name
    save_path = out_dir + '/' + wav_name
    #print(vad)
    write_wave(save_path, voiced_frames, sample_rate)

if __name__ == "__main__":
    in_wave = './16kHz_1ch_442.wav'
    outwav='./'
    filter(in_wave, outwav, expand=False)