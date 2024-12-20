import nltk, re, pprint
from nltk import word_tokenize
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import scipy

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

window_size = 2 #How many words in sequence to consider to be in the window
# Create a list of co-occurring word pairs
co_occurrences = defaultdict(Counter)
for i, word in enumerate(words):
    for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
        if i != j:
            co_occurrences[word][words[j]] += 1

# Create a list of unique words
unique_words = list(set(words))
# Initialize the co-occurrence matrix
co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)

print(co_matrix_df)

#Convert the above matrix to sparse representation, saves memory
print(scipy.sparse.csr_matrix(co_matrix_df))