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
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tag import StanfordPOSTagger
#from sentiment_analysis_spanish import sentiment_analysis
#import tensorflow as tf
import timeit
from statistics import mean

t0 = time.time()

path_folder = r'C:\Users\32214609\OneDrive - Anheuser-Busch InBev\My Documents\audios_sac'

os.chdir(path_folder)


#%%

filenames = next(os.walk(path_folder), (None, None, []))[2]

#%%
r = sr.Recognizer()

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
                text = r.recognize_google(audio_listened, language='es-PE')
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}."+'\n'
                print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text
#%%

c = ''
d = []
t = []

for i in filenames:
    start = timeit.default_timer()
    whole_text = get_large_audio_transcription(i)
    stop = timeit.default_timer()
    print(whole_text)
    output_file = open(f'C:/Users/32214609/OneDrive - Anheuser-Busch InBev/My Documents/files_sac/{i}.txt','w')
    output_file.write(whole_text)
    output_file.write('\n')
    output_file.close()
    #np.savetxt('C:/Users/32214609/OneDrive - Anheuser-Busch InBev/Command_CenterPE_TLV/GESTION_GROW_CARE_CALIDAD/PROJ_SPEAK_UP/files/' + i + '.txt', whole_text, fmt='%s')
    c += whole_text
    d.append(whole_text)
    t.append(stop - start)
    
d_2 = pd.DataFrame(d)
d_2.rename(columns={0:'conversacion'}, inplace=True)


#%%
max_temp = max(t)
min_temp = min(t)
prom_temp = mean(t)
sum_temp = sum(t)/3600
cant_llamadas = len(t)

#%%

'''sentiment = sentiment_analysis.SentimentAnalysisSpanish()

for k in d_2['conversacion']:
    print(sentiment.sentiment(k))
    '''


#%%

t1 = time.time()

total = t1-t0

#%%

# Aquí empieza el tratamiento en nltk.

quitar_2 = {'\xc3\xa1':'a','\xc3\xa9':'e','\xc3\xad':'i','\xc3\xb3':'o','\xc3\xba':'u',
             '\xc3\x81':'A','\xc3\x89':'E','\xc3\x8d':'I','\xc3\x93':'O','\xc3\x9a':'U',
             '\xc3\x9c':'U','\xc3\xbc':'u',
             '\xc3\xb1':'n','\xc3\x91':'N',
             'Â':''}

from nltk.corpus import stopwords

replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
    }

spanish_stopwords = stopwords.words('spanish')

delete = ['señorita', 'bien', 'momentito', 'muchas', 'gracias', 'tan', 'usted',
          'va', 'entonces', 'ay', 'aqui', 'ahi', '$', 'se', 'ora', 'as', 'caballero', 'whisky','alo', 'pues',
          '20', 'señora', 'ma', 'ana']
spanish_stopwords = pd.DataFrame(spanish_stopwords)
delete = pd.DataFrame(delete)
spanish_stopwords = pd.concat([spanish_stopwords, delete])
spanish_stopwords = spanish_stopwords[0].replace(replacements,regex=True)
spanish_stopwords = spanish_stopwords.values.tolist()


#%%

# Limpieza del string.

mytext_clean = c.replace('á', 'a')
mytext_clean = mytext_clean.replace('é', 'e')
mytext_clean = mytext_clean.replace('í', 'i')
mytext_clean = mytext_clean.replace('ó', 'o')
mytext_clean = mytext_clean.replace('ú', 'u')
mytext_clean = mytext_clean.replace('.','')
mytext_clean = mytext_clean.lower()
mytext_clean = mytext_clean.replace('\d+', '')
mytext_clean = mytext_clean.replace('[^\w\s]', '')
mytext_clean = mytext_clean.replace('\d', '')
mytext_clean = mytext_clean.replace('\\n', '')

mytext_clean_list = mytext_clean.split()

mytext_clean_list_stopwords = [word for word in mytext_clean_list if word not in spanish_stopwords]
space = " "
mytext_clean_list_stopwords_string = space.join(mytext_clean_list_stopwords)

tokenized = nltk.word_tokenize(mytext_clean_list_stopwords_string)

#%%

# Análisis descriptivo.

freq_dist = nltk.FreqDist(tokenized)
freq_dist.most_common(100)

text_all_words = nltk.Text(tokenized)
#freq_dist_bigrams = nltk.FreqDist(nltk.bigrams(text_all_words))
freq_dist = nltk.FreqDist((text_all_words))

most_common_20 = list(map(lambda x: list(x), freq_dist.most_common(30)))
df = pd.DataFrame(most_common_20, columns=['word', 'count'])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot()

sns.barplot(y=df['word'], x=df['count'], ax=ax, palette="RdYlBu")
ax.set_title("Frecuencia de las 20 palabras más comunes", pad=20);
ax.set_xlabel("Count")
ax.set_ylabel("")
plt.show()

#%%

# Análisis de sentimiento - Tratamiento.

c_2 = ''.join(filter(lambda x: not x.isdigit(),mytext_clean_list_stopwords_string))
c_3 = c_2.replace('$', '')
output_file_2 = open(r'C:\Users\32214609\OneDrive - Anheuser-Busch InBev\My Documents\files_sac\consolidado.txt','w')
output_file_2.write(c_3)
output_file_2.write('\n')
output_file_2.close()
#tokenized_2 = nltk.word_tokenize(c_2)
