import librosa
import tensorflow as tf
import numpy as np

def preprocessing (audio):
    spectrograms = []
    X, sr = librosa.load(audio)
    X_trim = librosa.effects.trim(X,top_db=35)
    if X_trim[0].shape[0] >= 65000:
        X_final = X_trim[0][:65000]
        X_final = tf.convert_to_tensor(X_final).numpy()
    else:
        zero_padding = tf.zeros([65000]-tf.shape(X_trim[0]),dtype=tf.float32)
        X_final = tf.concat([X_trim[0],zero_padding],0).numpy()
    S = librosa.feature.melspectrogram(y=X_final, sr=sr,n_mels=128)
    spectrograms.append(S)
    X= np.array(spectrograms)
    X_flat=X.reshape(X.shape[0],128*127)
    return X_flat