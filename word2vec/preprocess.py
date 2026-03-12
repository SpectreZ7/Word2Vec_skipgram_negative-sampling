
from pathlib import Path
# load the data 
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text
text = load_data('word2vec\\dataset\\text8.txt')

# tokenise data into words
def tokenise(text):
    return text.split()
tokens = tokenise(text)

# get word count and frequencies
from collections import Counter
def get_word_count(tokens):
    word_count = Counter(tokens)
    return word_count
word_count = get_word_count(tokens)

#filter rare words to improve efficiency and rare words probably won't be trained well anyway
min_count = 5
vocab = [w for w, c in word_count.items() if c >= min_count]
vocab = sorted(vocab)
print(len(get_word_count(vocab)))


#now create word and id mappings
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for i, w in word_to_id.items()}

#now convert tokens to ids

token_ids = [word_to_id[w] for w in tokens if w in word_to_id]

#make an array
import numpy as np
token_ids = np.array(token_ids, dtype=np.int32)#

#save vocab

import json

with open("vocab.json", "w") as f:
    json.dump(word_to_id, f)