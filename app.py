import streamlit as st
import librosa
from audio_recorder_streamlit import audio_recorder
import pickle
import io
import numpy as np
import tensorflow as tf
import base64
import webbrowser
import speech_recognition as sr





st.title("Silvertone!")
#st.header("Record a 5 seconds audio, and receive a % ....")
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.sidebar.title("Welcome to Silvertone!")
st.sidebar.image("logo_new.png", use_column_width=True)
st.sidebar.write("Contributors:")
st.sidebar.write("[Luiz Lianza](https://github.com/lalianza)")
st.sidebar.write("[Victor Sattamini](https://github.com/vsattamini)")
st.sidebar.write("[Lucas Gama](https://github.com/lucasgama1207)")
st.sidebar.write("[Guilherme Barreto](https://github.com/guipyc)")


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
r = sr.Recognizer()

if audio_bytes:
    st.audio(audio_bytes, format='audio/wav')
    X_flat = preprocessing(io.BytesIO(audio_bytes))
    model_pickle = open("model_four_emotions_72,96.sav", "rb")
    model = pickle.load(model_pickle)
    y = model.predict(X_flat)
    st.subheader(y[0])
    audio_source = sr.AudioData(audio_bytes,44100,4)
    try:
        text = r.recognize_google(audio_data=audio_source, language = 'en', show_all = True )
        st.subheader(text['alternative'][0]["transcript"])
    except:
        text = "Sorry, can you repeat that?"
        st.subheader(text)



