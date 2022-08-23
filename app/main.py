import random
from typing import Union
import json
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
with open('./models/intents.json', encoding="utf8") as f:
    intents = json.load(f)

words = pickle.load(open('./models/words.pkl','rb'))
classes = pickle.load(open('./models/classes.pkl','rb'))
model = load_model('./models/chatbotmodel.h5')

def clean_up_sentence (sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key= lambda  x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return  return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

class Chat(BaseModel):
    text: str

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
def chat_with_bot(chat: Chat):
    message = chat.text.lower()
    ints = predict_class(message)
    res = get_response(ints, intents)
    return { "chat": res }
