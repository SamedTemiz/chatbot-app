import numpy as np
import nltk

from trnlp import TrnlpToken
from snowballstemmer import TurkishStemmer

stemmer = TurkishStemmer()
tokenizer = TrnlpToken()

def tokenize(sentence):
    tokenizer.settext(sentence)
    return tokenizer.wordtoken


def stem(word):
    #stemmer.setword(word)

    #return stemmer.get_stem
    return stemmer.stemWord(word.lower())


def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag


sentence = "Hangi ürünlere sahipsiniz?"
tokenized_sentence = tokenize(sentence)
print(tokenized_sentence)

stemmed_words = [stem(w) for w in tokenized_sentence]
print(stemmed_words)