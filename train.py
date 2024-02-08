import numpy as np
import json
from text_preprocess import tokenize, lemmatize, bag_of_words

with open('Intents/intents.json') as f:
    data = json.load(f)

tags = []  # list of tags
tokens = []  # list of all the words, punctuations and numbers
tags_with_tokens = []

for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        token = tokenize(pattern)
        tokens.extend(token)

        tags_with_tokens.append((tag, token))

ignore_characters = ['?', '!', '.', ',', '-']
lemmatized_tokens = [lemmatize(i) for i in tokens if i not in ignore_characters]
unique_tags = sorted(set(tags))  # remove duplicates from tags
unique_tokens = sorted(set(lemmatized_tokens))  # remove duplicates from tokens

X_train = []
y_train = []

for (tag, pattern_sentence) in tags_with_tokens:

    label = tags.index(tag)
    y_train.append(label)

    lemmatized_pattern_sentence = [lemmatize(i) for i in pattern_sentence]
    X_train.append(bag_of_words(lemmatized_pattern_sentence, unique_tokens))

X_train = np.array(X_train)
y_train = np.array(y_train)


print(tags_with_tokens)
print(unique_tokens)
print(X_train)
print(y_train)