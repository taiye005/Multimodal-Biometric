import librosa
import numpy as np

audio_file = "voice1.wav"

# Function to extract features from an audio file
def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean


audio_files = ['voice1.wav']  
for i, audio_file in enumerate(audio_files, start=1):
    features = extract_audio_features(audio_file)
    print(f'Feature vector for audio {i}: {features}')


    