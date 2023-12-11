import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """
    cümleleri kelimelere ayirip token haline getiriyoruz
    noktalama, numara ve kelime hepsi ayriliyor
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = kelimenin kökünü buluyoruz
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    bölünen kelimeleri temsil eden kelimeler var mi diye kontrol ediyoruz
    bunu vektör kullanarak 1-0 ile yapiyoruz

    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag


words = ["Organize", "Organizeler", "Organize etmek"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)