import pickle
import numpy as np
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
    tags.append(tag.lower())

    for pattern in intent['patterns']:
        token = tokenize(pattern)
        tokens.extend(token)

        tags_with_tokens.append((tag, token))

lemmatized_tokens = [lemmatize(i) for i in tokens if i not in ignore_characters]
unique_tags = sorted(set(tags))  # remove duplicates from tags
unique_tokens = sorted(set(lemmatized_tokens))  # remove duplicates from tokens

pickle.dump(unique_tags, open('tags', 'wb'))
pickle.dump(unique_tokens, open('tokens', 'wb'))

training_data = []

for (tag, pattern_sentence) in tags_with_tokens:
    tag_label = np.zeros(len(unique_tags))
    tag_label[unique_tags.index(tag)] = 1

    lemmatized_pattern_sentence = [lemmatize(i) for i in pattern_sentence]
    bag = bag_of_words(lemmatized_pattern_sentence, unique_tokens)

    training_data.append(np.concatenate([tag_label, bag]))

training_data = np.array(training_data)
np.random.seed(12)  # set a fixed random value
np.random.shuffle(training_data)

X_train = training_data[:, 6:]
y_train = training_data[:, :6]

model = Sequential()
model.add(Dense(128, input_shape=(len(unique_tokens),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(68, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(unique_tags), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000, batch_size=8, verbose=1)
model.save('chatbot_model')

print('Training is done')
