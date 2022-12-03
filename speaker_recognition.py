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

os.chdir(r'C:\Users\32214609\OneDrive - Anheuser-Busch InBev\Command_CenterPE_TLV\GESTION_GROW_CARE_CALIDAD\PROJ_SPEAK_UP\audios_lima')

#%%

filename = '2022111012334187104803847837416701.wav'

#%%

from resemblyzer import preprocess_wav, VoiceEncoder

wav = preprocess_wav(filename)
encoder = VoiceEncoder("cpu")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
print(cont_embeds.shape)

#REF: https://github.com/resemble-ai/Resemblyzer, https://medium.com/saarthi-ai/who-spoke-when-build-your-own-speaker-diarization-module-from-scratch-e7d725ee279

#%%

from spectralcluster import SpectralClusterer

clusterer = SpectralClusterer(
    min_clusters=2,
    max_clusters=100,
    max_iter=100)

labels = clusterer.predict(cont_embeds)

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

r = sr.Recognizer()

#%%

# a function that splits the audio file into chunks
# and applies speech recognition
def get_large_audio_transcription_new(path,label):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    '''sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )'''
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(label, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        #audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened, language='es-MX')
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}."+'\n'
                print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text

#%%

# a function that splits the audio file into chunks
# and applies speech recognition
def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened, language='es-MX')
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}."+'\n'
                print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text

#%%

whole_text = get_large_audio_transcription(filename)
print(whole_text)

#%%

whole_text_new = get_large_audio_transcription_new(filename)
print(whole_text_new)


#%%

'''def transcribe_file_with_diarization(speech_file):

    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()

    encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16
    sample_rate_hertz=48000
    language_code='es-MX'
    enable_speaker_diarization=True
    enable_automatic_punctuation=True
    diarization_speaker_count=4

    config = {
        "encoding": encoding,
        "sample_rate_hertz": sample_rate_hertz,
        "language_code": language_code,
        "enable_speaker_diarization": enable_speaker_diarization,
        "enable_automatic_punctuation": enable_automatic_punctuation,
        # Optional:
        "diarization_speaker_count": diarization_speaker_count
    }

    print('Waiting for operation to complete…')
    response = client.recognize(config, audio)

    # The transcript within each result is separate and sequential per result.
    # However, the words list within an alternative includes all the words
    # from all the results thus far. Thus, to get all the words with speaker
    # tags, you only have to take the words list from the last result:

    result = response.results[-1]
    words_info = result.alternatives[0].words

    speaker1_transcript=''
    speaker2_transcript=''
    speaker3_transcript=''
    speaker4_transcript=''

    # Printing out the output:
    for word_info in words_info:
        if(word_info.speaker_tag==1): 
            speaker1_transcript=speaker1_transcript+word_info.word+' '
        if(word_info.speaker_tag==2): 
            speaker2_transcript=speaker2_transcript+word_info.word+' '
        if(word_info.speaker_tag==3): 
            speaker3_transcript=speaker3_transcript+word_info.word+' '
        if(word_info.speaker_tag==4): 
            speaker4_transcript=speaker4_transcript+word_info.word+' '

    print("speaker1: '{}'".format(speaker1_transcript))
    print("speaker2: '{}'".format(speaker2_transcript))
    print("speaker3: '{}'".format(speaker3_transcript))
    print("speaker4: '{}'".format(speaker4_transcript))'''

