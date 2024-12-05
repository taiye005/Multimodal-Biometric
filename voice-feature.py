import librosa
import numpy as np

def extract_audio_features(file_path):
   
    print(f"Loading audio file: {file_path}")
   
    y, sr = librosa.load(file_path, sr=None)
   
    print(f"Audio loaded, sample rate: {sr}, length: {len(y)}")
   
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
   
    print("MFCCs extracted")
    return mfccs.flatten()

audio_file = 'voice3.wav'

features = extract_audio_features(audio_file)

print("Voice feature vector:", features)


