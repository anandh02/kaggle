import numpy as np

embeddings_index = dict()
f = open('Glove+Wiki/glove.6B.50d.txt', encoding="utf8")
for line in f:
    values = line.split()
    print(values)
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

