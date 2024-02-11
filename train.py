import numpy as np
import random
import json
from text_preprocessor import tokenize, lemmatize, bag_of_words
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

with open('Intents/intents.json') as f:
    data = json.load(f)

tags = []  # list of tags
tokens = []  # list of all the words, punctuations and numbers
tags_with_tokens = []
ignore_characters = ['?', '!', '.', ',', '-']

for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        token = tokenize(pattern)
        tokens.extend(token)

        tags_with_tokens.append((tag, token))

lemmatized_tokens = [lemmatize(i) for i in tokens if i not in ignore_characters]
unique_tags = sorted(set(tags))  # remove duplicates from tags
unique_tokens = sorted(set(lemmatized_tokens))  # remove duplicates from tokens

training_data = []

for (tag, pattern_sentence) in tags_with_tokens:
    label = tags.index(tag)

    lemmatized_pattern_sentence = [lemmatize(i) for i in pattern_sentence]
    bag = bag_of_words(lemmatized_pattern_sentence, unique_tokens)

    training_data.append(np.concatenate(([label], bag)))

training_data = np.array(training_data)
np.random.seed(12)  # set a fixed random value
np.random.shuffle(training_data)

X_train = training_data[:, 1:]
y_train = training_data[:, 0]

model = Sequential()
model.add(Dense(128, input_shape=len(unique_tokens), activation='relu'))
model.add(Dropout(0.5))


print(training_data)
print(y_train)
print(X_train)


print(len(X_train[0]))
print(len(unique_tokens))