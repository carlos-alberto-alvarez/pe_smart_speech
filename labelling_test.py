import speech_recognition as sr
import os
import glob
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import time
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import wave
import time

os.chdir(r'C:\Users\32214609\OneDrive - Anheuser-Busch InBev\My Documents\audios_sac')

#%%

filename = 'America Johana Gonzalez Mendez2.wav'

#%%

from resemblyzer import preprocess_wav, VoiceEncoder

wav = preprocess_wav(filename)
encoder = VoiceEncoder("cpu")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
print(cont_embeds.shape)

#REF: https://github.com/resemble-ai/Resemblyzer, https://medium.com/saarthi-ai/who-spoke-when-build-your-own-speaker-diarization-module-from-scratch-e7d725ee279

#%%

from sklearn.cluster import SpectralClustering

#clustering = SpectralClustering(eigen_solver = 'lobpcg', n_components=100, n_clusters=2, affinity = 'nearest_neighbors', n_neighbors=200).fit(cont_embeds)

clustering = SpectralClustering(n_clusters=2, affinity = 'nearest_neighbors', n_neighbors=200).fit(cont_embeds)


labels = clustering.fit_predict(cont_embeds)

#%%

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward') 
labels=cluster.fit_predict(cont_embeds)

#%%

def create_labelling(labels,wav_splits):
    from resemblyzer.audio import sampling_rate
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0

    for i,time in enumerate(times):
        if i>0 and labels[i]!=labels[i-1]:
            temp = [str(labels[i-1]),start_time,time]
            labelling.append(tuple(temp))
            start_time = time
        if i==len(times)-1:
            temp = [str(labels[i]),start_time,time]
            labelling.append(tuple(temp))

    return labelling
  
labelling = create_labelling(labels,wav_splits)

#%%

label_speaker = [lis[0] for lis in labelling]
label_start_time = [lis[1] for lis in labelling]
label_finish_time = [lis[2] for lis in labelling]


#%%

sound = AudioSegment.from_wav(filename)
# Time to milliseconds conversion
StrtTime = [item * 1000 for item in label_start_time]
EndTime = [item * 1000 for item in label_finish_time]

for i,j in zip(StrtTime, EndTime):
    debug = sound[i:j]
    debug.export(fr'C:\Users\32214609\OneDrive - Anheuser-Busch InBev\My Documents\audios_sac\test_cluster\test{i}.wav', format="wav")