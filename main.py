import nltk
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')

stemmer = PorterStemmer()


# split a string into units(words, punctuations, numbers)
def tokenization(sentence):
    return nltk.word_tokenize(sentence)

# generate the root from the words.
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

text = "Sad to see you go :("
print(text)
print(tokenization(text))
print([stem(i) for i in ['Organize', 'organizing', 'oRGANIZED']])

