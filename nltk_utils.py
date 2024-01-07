import numpy as np
import nltk
from trnlp import TrnlpToken
from snowballstemmer import TurkishStemmer

import re

# Türkçe kök bulma işlemi için stemmer ve tokenizer tanımlanır
stemmer = TurkishStemmer()
tokenizer = TrnlpToken()

# Cümleyi tokenize eden fonksiyon tanımlanır
def tokenize(sentence):
    tokenizer.settext(sentence)
    return tokenizer.wordtoken

# Kelimeyi köküne çeviren fonksiyon tanımlanır
def stem(word):
    # Verilen kelimenin kökünü bulur ve küçük harfe dönüştürür
    return stemmer.stemWord(word.lower())

# Cümledeki kelimeleri köklerine göre ayıran ve bir dizi haline getiren fonksiyon
def bag_of_words(tokenized_sentence, words):
    # Her bir kelimeyi köküne çevir
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # Bag of words vektörünü başlangıçta sıfırlarla doldur
    bag = np.zeros(len(words), dtype=np.float32)
    
    # Her bir kelimenin, tüm kelimeler listesinde olup olmadığını kontrol eder
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

# Sayısal değerleri tokenize etmek için fonksiyon
def tokenize_number(number):
    tokens = re.finditer(r"\d+", number)
    return [token.group() for token in tokens]




# Örnek bir cümle tanımlanır ve bu cümle tokenlara ayrılır
sentence = "12GB ram 256gb ssd ve HP marka laptop istiyorum"

tokenized_number = tokenize_number(sentence)
print("Numaralar: ", tokenized_number)

tokenized_sentence = tokenize(sentence)
print("Tokenize edilmiş kelimeler: ", tokenized_sentence)

# Tokenize edilmiş kelimelerin her biri köküne çevrilir ve ekrana yazdırılır
stemmed_words = [stem(w) for w in tokenized_sentence]
print("Kelime kökleri: ", stemmed_words)
