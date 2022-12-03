import speech_recognition as sr
import pyaudio
import pyttsx3
import nltk
import pandas as pd

#%%
 
# Initialize the recognizer
r = sr.Recognizer()
 
# Function to convert text to
# speech
def SpeakText(command):
     
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()
     
#%%

# Loop infinitely for user to
# speak
 
while(1):   
     
    # Exception handling to handle
    # exceptions at the runtime
    try:
         
        # use the microphone as source for input.
        with sr.Microphone() as source2:
             
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.2)
             
            #listens for the user's input
            audio2 = r.listen(source2)


             
            # Using google to recognize audio
            MyText = r.recognize_google(audio2, language='es-MX')
            MyText = MyText.lower()
 
            print(MyText)
            break
             
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
         
    except sr.UnknownValueError:
        print("unknown error occured")

#%%

# Aquí empieza el nltk

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

spanish_stopwords = pd.DataFrame(spanish_stopwords)
spanish_stopwords = spanish_stopwords[0].replace(replacements,regex=True)
spanish_stopwords = spanish_stopwords.values.tolist()


#%%
mytext_clean = MyText.replace('á', 'a')
mytext_clean = mytext_clean.replace('é', 'e')
mytext_clean = mytext_clean.replace('í', 'i')
mytext_clean = mytext_clean.replace('ó', 'o')
mytext_clean = mytext_clean.replace('ú', 'u')

mytext_clean_list = mytext_clean.split()

mytext_clean_list_stopwords = [word for word in mytext_clean_list if word not in spanish_stopwords]
space = " "
mytext_clean_list_stopwords_string = space.join(mytext_clean_list_stopwords)

tokenized = nltk.word_tokenize(mytext_clean_list_stopwords_string)

