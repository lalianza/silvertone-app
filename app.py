import streamlit as st
import requests
import pandas as pd
import numpy as np
import librosa
from audio_recorder_streamlit import audio_recorder
from audiorecorder import audiorecorder
import pickle
import IPython.display as ipd
from scipy.io.wavfile import write
import pickle
import io

#from io import BytesIO
#import urllib
#from pydub import AudioSegment


#st.title("Audio Recorder")
#audio = audiorecorder("Click to record", "Recording...")
#
#if len(audio) > 0:
#    # To play audio in frontend:
#    st.audio(audio)
#
#    # To save audio to a file:
#    wav_file = open("audio.mp3", "wb")
#    wav_file.write(audio.tobytes())
#    #sample, sr = librosa.load(audio)
#    #ipd.Audio(sample)
#    #sample,sr = librosa.core.load('audio.mp3')
#    #ipd.Audio(sample)

audio_bytes = audio_recorder()
if audio_bytes:
    audio = st.audio(audio_bytes, format='audio/wav')
    # filename = 'audio.pkl'
# with open('audio.pkl', 'rb') as f:
#     audio_lib = pickle.load(f)
data, samplerate = librosa.load(io.BytesIO(audio))
    # print(len(data))
    # print(type(data))
    #pickle.dump(audio_bytes, open(filename, 'wb'))
#    wav_file = open("audio.mp3", "wb")
#    wav_file.write(audio_bytes.tobytes())
    #sample,sr = librosa.core.load(audio_bytes)
    #ipd.Audio(sample)
    #audio = st.audio(audio_bytes, format="audio/wav")
    #st.markdown(audio_bytes)
    #scipy.io.wavfile.write(filename, rate, data)
    #filename = 'audio.wav'
    #pickle.dump(audio_bytes, open(filename, 'wb'))
    #urllib.request.urlretrieve('http://localhost:8501/media/ea89cb71a0175954e7bbaf5c2a00fec7717d2268080a96b03b3e24f8.wav', 'audio.wav')
    #ipd.Audio(filename)
    #ipd.Audio('audio.wav'[5])



#sample,sr = librosa.core.load(audio.wav)
#iipd.Audio(sample)
#
'''



## Once we have these, let's call our API in order to retrieve a prediction

See ? No need to load a `model.joblib` file in this app, we do not even need to know anything about Data Science in order to retrieve a prediction...

🤔 How could we call our API ? Off course... The `requests` package 💡
'''

#url = 'https://taxifare.lewagon.ai/predict'


#response = requests.get(url,params).json()
#response


#if url == 'https://taxifare.lewagon.ai/predict':

 #   st.markdown('Maybe you want to use your own API for the prediction, not the one provided by Le Wagon...')

'''

2. Let's build a dictionary containing the parameters for our API...

3. Let's call our API using the `requests` package...

4. Let's retrieve the prediction from the **JSON** returned by the API...

## Finally, we can display the prediction to the user
'''
