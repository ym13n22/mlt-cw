import nltk, re, pprint
from nltk import word_tokenize

with open("text8/text8","r")as f:
    data =f.readline()

type(data)
print(type(data))
print(len(data))
print(data[:75])
tokens=word_tokenize(data)
type(tokens)
print(type(tokens))
print(len(tokens))
print(tokens[:10])
text=nltk.Text(tokens)
print(type(text))
print(len(text))
print(text[:10])

#build a vocabulary
words=[w.lower() for w in tokens]
vocab=set(set(words))
print(len(vocab))