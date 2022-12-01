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
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.graph_objs import *
import librosa.display


st.title("Silvertone")
st.markdown("""We created this app to be able to recognize emotion in spoken english.
We hope you enjoy it
To use it:
1. Use the button below to record an audio acting out an emotion (try to keep it short, the app likes it better that way!)
2. Let the app do its magic
3. See if it matches! 
4. The spectrogram tab will give you a visual representation of your audio""")
st.subheader("Record an audio.... :")
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

#st.sidebar.title("Welcome to Silvertone!")
st.sidebar.image("final_logo.png", use_column_width=True)
st.sidebar.write("[Repository](https://github.com/vsattamini/silvertone)")

st.sidebar.write("Contributors:")
st.sidebar.write("[Luiz Lianza](https://github.com/lalianza)")
st.sidebar.write("[Victor Sattamini](https://github.com/vsattamini)")
st.sidebar.write("[Lucas Gama](https://github.com/lucasgama1207)")
st.sidebar.write("[Guilherme Barreto](https://github.com/guipyc)")


def preprocessing (audio):
    mfcc = []
    X, sr = librosa.load(audio)
    X_trim = librosa.effects.trim(X,top_db=35)
    if X_trim[0].shape[0] >= 50000:
        X_final = X_trim[0][:50000]
        X_final = tf.convert_to_tensor(X_final).numpy()
    else:
        zero_padding = tf.zeros([50000]-tf.shape(X_trim[0]),dtype=tf.float32)
        X_final = tf.concat([X_trim[0],zero_padding],0).numpy()
    S = librosa.feature.mfcc(y=X_final, sr=sr)
    mfcc.append(S)
    X= np.array(mfcc)
    X_flat=X.reshape(X.shape[0],20*98)
    return X_flat

filename = None

audio_bytes = audio_recorder()
r = sr.Recognizer()

if audio_bytes:
    st.audio(audio_bytes, format='audio/wav')
    X_flat = preprocessing(io.BytesIO(audio_bytes))
    model_pickle = open("model_svc_5emo_64,5.sav", "rb")
    model = pickle.load(model_pickle)
    y = model.predict(X_flat)
    st.subheader(y[0])
    audio_source = sr.AudioData(audio_bytes,44100,4)
    try:
        text = r.recognize_google(audio_data=audio_source, language = 'en', show_all = True )
        st.subheader(text['alternative'][0]["transcript"].title())
    except:
        text = "Sorry, can you repeat that?"
        st.subheader(text)
    tab1, tab2 = st.tabs([" ","Spectrogram"])
    with tab2:
       fig, ax = plt.subplots(facecolor=(0, 0, 0,0))
       X, sr = librosa.load((io.BytesIO(audio_bytes)))
       #S_dB = librosa.power_to_db(y=X,sr=sr,n_mels=128, ref=np.max)
       S = librosa.feature.melspectrogram(y=X, sr=sr, n_mels=128)
       S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
       #mel_spec = librosa.features.melspectogram
       img = librosa.display.specshow(S_db_mel, x_axis='time',
                                y_axis='mel', sr=sr,
                                fmax=8000, ax=ax)
       fig.colorbar(img, ax=ax, format='%+2.0f dB')
       #ax.set_facecolor(0,0,0,0)
       ax.set(title='Mel-frequency spectrogram')
       st.pyplot(fig)



