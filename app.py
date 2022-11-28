import streamlit as st
import librosa
from audio_recorder_streamlit import audio_recorder
import pickle
import io
import numpy as np
import pandas as pd
import tensorflow as tf

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

filename = None

audio_bytes = audio_recorder()

if audio_bytes:
    st.audio(audio_bytes, format='audio/wav')
    print(type(audio_bytes))
    X_flat = preprocessing(io.BytesIO(audio_bytes))
    model_pickle = open("54,6_model.sav", "rb")
    model = pickle.load(model_pickle)
    y = model.predict(X_flat)
    if y == "01":
        response = "Neutral"
    elif y == "02":
        response = "Calm"
    elif y == "03":
        response = "Happy"
    elif y == "04":
        response = "Sad"
    elif y == "05":
        response = "Angry"
    elif y == "06":
        response = "Fearful"
    elif y == "07":
        response = "Disgust"
    elif y == "08":
        response = "Surprised"
    else:
        response = "Error"
    st.text(response)

        
    
    
 
    # data, samplerate = librosa.load(io.BytesIO(audio_bytes))
    # st.text(len(data))
    # st.text(type(data))

