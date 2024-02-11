import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


# split a string into units(words, punctuations, numbers)
def tokenize(sentence):
    return word_tokenize(sentence)


# generate the root from the words.
def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words))
    for (index, word) in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1

    return bag



