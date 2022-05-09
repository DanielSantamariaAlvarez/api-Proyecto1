import json
from plistlib import load
from DataModel import DataModel, DListar
from pandas import json_normalize
from joblib import load
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
from string import punctuation
from nltk.corpus import stopwords
import re  # regular expression
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('omw-1.4')
from os.path import dirname, join, realpath
import joblib



#######

import re, string, unicodedata
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from fastapi.middleware.cors import CORSMiddleware



#######

app = FastAPI()

# Arreglo de las urls que pueden acceder al backend
origins = ["http://localhost:8000", "https://proyecto1-etapa2-bi-front.herokuapp.com"]

# Manejo de las cors para habilitar los endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def stemw(palabras):
    stemmer = LancasterStemmer()
    stems = []
    for palabra in palabras:
        stem = stemmer.stem(palabra)
        stems.append(stem)
    return stems

def lemmatizew(palabras):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for palabra in palabras:
        lemma = lemmatizer.lemmatize(palabra, pos='v')
        lemmas.append(lemma)
    return lemmas

def stemlemmatize(palabras):
    stems = stemw(palabras)
    lemmas = lemmatizew(palabras)
    return stems + lemmas

def uselessdata(words):
    dot = words.index('.')
    new_words = words[dot+1:]
    return new_words

def nonascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def numbers(words):
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def removestopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def preprocessing(words):
    words = uselessdata(words)
    words = lowercase(words)
    words = numbers(words)
    words = punctuation(words)
    words = nonascii(words)
    words = removestopwords(words)
    return words


@app.get("/")
def read_root():
    return{
        "Proyecto": "1 - Parte 2",
        "Integrante 1": "Daniel Santamaría Álvarez",
        "Integrante 2": "Laura Daniela Manrique",    
        "Integrante 3": "Gabriel Serna"   
    }



@app.post("/knn")
def postKNN(data: DListar):
    dict = jsonable_encoder(data)
    df = json_normalize(dict['texto']) 
    df.columns = DataModel.columns()
    model = load("./pipelines/pipelineKNN.joblib")
    result = model.predict(df)
    lists = result.tolist()
    json_predict = json.dumps(lists)
    return {"predict": json_predict}


@app.post("/nb")
def postNB(data: DListar):
    dict = jsonable_encoder(data)
    df = json_normalize(dict['texto']) 
    df.columns = DataModel.columns()
    model = load("./pipelines/pipelineNB.joblib")
    result = model.predict(df)
    lists = result.tolist()
    json_predict = json.dumps(lists)
    return {"predict": json_predict}


@app.post("/rl")
def postRL(data: DListar):
    dict = jsonable_encoder(data)
    df = json_normalize(dict['texto']) 
    df_clean = limpiar(df)
    model = load("./pipelines/pipelineNBs.joblib")
    result = model.predict(df_clean)
    lists = result.tolist()
    json_predict = json.dumps(lists)
    return {"Predict": json_predict, "Cleaned": df}


def limpiar(df_t):
    df_t['study_and_condition'] = df_t['study_and_condition'].apply(contractions.fix) #Aplica la corrección de las contracciones 
    df_t['study_and_condition'] = df_t['study_and_condition'].apply(word_tokenize).apply(preprocessing) #Aplica la eliminación del ruido
    df_t['study_and_condition'] = df_t['study_and_condition'].apply(stemlemmatize) #Aplica la normalización
    df_t['study_and_condition'] = df_t['study_and_condition'] .apply(lambda x: ' '.join(map(str, x)))
    return df_t




