import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH, N_MFCC

def extract_features_from_audio(file_path):
    """
    Loads an audio file and returns a feature matrix of shape (n_frames, n_features).
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    mfcc = librosa.feature.mfcc(
        y=y, 
        sr=sr, 
        n_mfcc=N_MFCC, 
        n_fft=FRAME_LENGTH, 
        hop_length=HOP_LENGTH
    )
    
    delta_mfcc = librosa.feature.delta(mfcc)
    
    rms = librosa.feature.rms(
        y=y, 
        frame_length=FRAME_LENGTH, 
        hop_length=HOP_LENGTH
    )
    
    features = np.vstack([mfcc, delta_mfcc, rms]).T
    return features