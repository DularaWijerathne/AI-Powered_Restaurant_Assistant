import pickle
import random
import json
import numpy as np
from nltk.tokenize import word_tokenize
from text_preprocessor import lemmatize, bag_of_words

with open('Intents/intents.json') as f:
    intents_json = json.load(f)

tags = pickle.load(open('tags.pkl', 'rb'))
tokens = pickle.load(open('tokens.pkl', 'rb'))
model = pickle.load(open('chatbot_model.pkl', 'rb'))


def predict_class(sentence):
    tokenized_sentence = word_tokenize(sentence)
    lemmatized_sentence = [lemmatize(i) for i in tokenized_sentence]
    token_bag = bag_of_words(lemmatized_sentence, tokens)

    likelihoods = model.predict(np.array([token_bag]))[0]  # output is given as a 2D array. use [0] to return a 1D array

    results = [[i, r] for i, r in enumerate(likelihoods)]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = [{'intent': tags[i[0]], 'probability': i[1]} for i in results]

    return return_list


def get_response(sentence, intents_json):
    THRESHOLD = 0.8
    intents_list = intents_json['intents']
    intent_probabilities = predict_class(sentence)[0]
    response = ''

    if intent_probabilities['probability'] > THRESHOLD:
        for i in intents_list:
            if i['tag'] == intent_probabilities['intent']:
                response = random.choice(i['responses'])
                break
    else:
        response = "Sorry, I can't understand"

    return response


def chat():
    print('talk with the chatbot')
    while True:
        message = input('you: ')
        response = get_response(message, intents_json)
        if message.lower() == 'quit':
            print('Chatbot quit')
            break
        print(response)


chat()
