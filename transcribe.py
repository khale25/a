from itertools import groupby
import numpy as np
import warnings
import base64
import torchaudio
import torch
import os
import io
from os import path
from pydub import AudioSegment
import librosa
import soundfile as sf
from pydub.silence import split_on_silence
import math

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '\\' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '\\' + split_filename, format="wav")
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')

class Decoder():
    def __init__(self, labels):
        self.labels = labels
        self.blank_idx = self.labels.index('_')
        self.space_idx = self.labels.index(' ')

    def process(self, probs, wav_len, word_align):
        assert len(self.labels) == probs.shape[1]
        for_string = []
        argm = torch.argmax(probs, axis=1)
        align_list = [[]]
        for j, i in enumerate(argm):
            if i == self.labels.index('2'):
                try:
                    prev = for_string[-1]
                    for_string.append('$')
                    for_string.append(prev)
                    align_list[-1].append(j)
                    continue
                except:
                    for_string.append(' ')
                    warnings.warn('Token "2" detected a the beginning of sentence, omitting')
                    align_list.append([])
                    continue
            if i != self.blank_idx:
                for_string.append(self.labels[i])
                if i == self.space_idx:
                    align_list.append([])
                else:
                    align_list[-1].append(j)

        string = ''.join([x[0] for x in groupby(for_string)]).replace('$', '').strip()

        align_list = list(filter(lambda x: x, align_list))

        if align_list and wav_len and word_align:
            align_dicts = []
            linear_align_coeff = wav_len / len(argm)
            to_move = min(align_list[0][0], 1.5)
            for i, align_word in enumerate(align_list):
                if len(align_word) == 1:
                    align_word.append(align_word[0])
                align_word[0] = align_word[0] - to_move
                if i == (len(align_list) - 1):
                    to_move = min(1.5, len(argm) - i)
                    align_word[-1] = align_word[-1] + to_move
                else:
                    to_move = min(1.5, (align_list[i+1][0] - align_word[-1]) / 2)
                    align_word[-1] = align_word[-1] + to_move

            for word, timing in zip(string.split(), align_list):
                align_dicts.append({'word': word,
                                    'start_ts': round(timing[0] * linear_align_coeff, 2),
                                    'end_ts': round(timing[-1] * linear_align_coeff, 2)})

            return string, align_dicts
        return string

    def __call__(self, probs, wav_len=0, word_align=False):
        return self.process(probs, wav_len, word_align)


class AudioTrans():
    def __init__(self, kwargs=None):
        self.device = torch.device('cpu')
        model_url = "https://models.silero.ai/models/en/en_v6.jit"
        self.model = self.load_model(model_url)
        self.decoder = Decoder(self.model.labels)


    def load_model(self, model_url):
        torch.set_grad_enabled(False)

        model_dir = os.path.join(os.path.dirname(__file__), "model")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, os.path.basename(model_url))

        if not os.path.isfile(model_path):
            print("Download pretrained stt model...")
            torch.hub.download_url_to_file(model_url, model_path, progress=True)

        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model


    def preprocess_input(self, input, target_sr=16000):
        # decode_string = base64.b64decode(input)
        # wav, sr = torchaudio.load(io.BytesIO(decode_string))
        wav, sr = torchaudio.load(input)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != target_sr:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav = transform(wav)
            sr = target_sr

        assert sr == target_sr
        return wav


    def process(self, input):
        if input is not None and isinstance(input, str):

            input = self.preprocess_input(input)

            output = self.model(input)
            transcribe = self.decoder(output[0].cpu())
            print("------trans-----")
            print(transcribe)
            return transcribe
        else:
            return None



m4a_file = '3_11_v1.m4a'
wav_filename = "3_11_v1.wav"

def convert_wav():
    track = AudioSegment.from_file(m4a_file,  format= 'm4a')
    file_handle = track.export(wav_filename, format='wav')

def split_audio():
    # x, sr = librosa.load(wav_filename, sr=16000)
    # #you will get x as audio file in numpy array and sr as original sampling rate
    # num = 50 
    # for i in range(0, len(x),num * sr):
    #     y = x[num * sr * i: num * sr *(i+1)]
    #     sf.write("dest_audio"+str(i)+".wav", y, sr)
    
    
    # sound_file = AudioSegment.from_wav(wav_filename)
    # audio_chunks = split_on_silence(sound_file, min_silence_len=5000, silence_thresh=-40 )
    
    # for i, chunk in enumerate(audio_chunks):
    #     out_file = "chunk{0}.wav".format(i)
    #     print("exporting", out_file)
    #     chunk.export(out_file, format="wav")
    pass


import whisper


if __name__ == "__main__":
    # s2t = AudioTrans()

    # split_audio()
    
    # s2t.process("C:\\Projects\\Text2Speech_Ha\\split\\0_3_11_v1.wav")
    #print()

    folder = 'C:\\Projects\\Text2Speech_Ha\\split'
    file = '3_11_v1.wav'
    split_wav = SplitWavAudioMubin(folder, file)
    split_wav.multiple_split(min_per_split=0.5)

    # model = whisper.load_model("large")
    # text = model.transcribe("C:\\Projects\\Text2Speech_Ha\\split\\0_3_11_v1.wav")

    # print(text['text'])
