import os
import numpy as np
from src.audio_features import extract_features_from_audio

def get_audio_files_from_dir(directory):
    """
    Recursively finds all audio files in the given directory and its subdirectories.
    """
    audio_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(('.wav', '.flac', '.mp3')) and not f.startswith('.'):
                audio_files.append(os.path.join(root, f))
    return sorted(audio_files)

def load_dataset(train_dir, max_files_per_class=35):
    """
    Loads features for speech (1) and noise (0) classes.
    """
    classes = {'noise': 0, 'speech': 1}
    X_list = []
    y_list = []
    
    for class_name, label_id in classes.items():
        class_folder = os.path.join(train_dir, class_name)
        if not os.path.exists(class_folder):
            raise FileNotFoundError(f"Folder not found: {class_folder}")
            
        all_files = get_audio_files_from_dir(class_folder)
        
        selected_files = all_files[:max_files_per_class]
        
        for file_path in selected_files:
            try:
                feats = extract_features_from_audio(file_path)
                X_list.append(feats)
                y_list.append(np.full(feats.shape[0], label_id))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y