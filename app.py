import streamlit as st
import librosa
from audio_recorder_streamlit import audio_recorder
import pickle
import io



filename = None

audio_bytes = audio_recorder()

if audio_bytes:
    st.audio(audio_bytes, format='audio/wav')
    print(type(audio_bytes))
    data, samplerate = librosa.load(io.BytesIO(audio_bytes))
    st.text(len(data))
    st.text(type(data))
