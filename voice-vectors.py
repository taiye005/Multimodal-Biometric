import librosa
import numpy as np

def extract_mfcc_features(audio_file, sr=16000, n_mfcc=20):
  """
  Extracts MFCC features from a given audio file.

  Args:
      audio_file (str): Path to the audio file.
      sr (int, optional): Sampling rate of the audio file. Defaults to 16000.
      n_mfcc (int, optional): Number of MFCC coefficients to extract. Defaults to 20.

  Returns:
      tuple: (mfcc_features, mfcc_delta)
          mfcc_features (numpy.ndarray): 2D array containing MFCC coefficients.
          mfcc_delta (numpy.ndarray): 2D array containing delta features (derivative of MFCCs).
  """
  # Load audio
  y, sr = librosa.load(audio_file, sr=sr)

  # Pre-emphasis (optional)
  # y = librosa.effects.preemphasis(y, sr=sr)

  # Mel spectrogram extraction
  melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

  # MFCC extraction
  mfcc_features = librosa.feature.mfcc(S=melspec, n_mfcc=n_mfcc)

  # Optional: Delta features (derivative of MFCCs)
  mfcc_delta = librosa.feature.delta(mfcc_features)

  return mfcc_features, mfcc_delta

# Example usage
audio_file = "voice1.wav"  
mfcc_features, mfcc_delta = extract_mfcc_features(audio_file)

# Print the shape of the features
print(mfcc_features.shape)
print(mfcc_delta.shape)

# You can now use these features for further processing or training your speech recognition model.