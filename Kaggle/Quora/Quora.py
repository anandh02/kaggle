import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
    from tqdm import tqdm

    from keras.preprocessing.text import Tokenizer
data = pd.read_csv('change2.csv')

data['target'].value_counts()

def word_count(ques):
    return len(ques.split())


data['word_count'] = data['question_text'].apply(word_count)
#
# data[['target','word_count']].groupby('target').mean().plot.bar()
# sns.boxplot('target','word_count',data = data)
# plt.show();


def cleaning(text):
    texts = ""

    words = [(word ) for word in text.split() if word not in stopwords.words('english')]
    # print(words)
    return " ".join(words)
# text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

# data['question_text'] = data['question_text'].apply(cleaning)
#
# data.to_csv('changed_data', sep='\t')

# writer_orig = pd.ExcelWriter('simple.xlsx', engine='xlsxwriter')
# data.to_excel(writer_orig, index=False, sheet_name='report')
# writer_orig.save()


embeddings_index = dict()
f = open('../Glove/Glove+Wiki/glove.6B.50d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(data['question_text'])

embedding_matrix = np.zeros((vocabulary_size, 50))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector







# def voca(sentence):
#     vocab = {}
#     for sent in tqdm(sentence):
#         for word in sent.split(' '):
# #             print(word)
#             try:
#                 vocab[word.lower()] += 1
#             except:
#                 vocab[word.lower()] = 1
#     return vocab
#
#
# vocab = voca(data['question_text'])
#
# def counter(sentence,embeddings_index):
#     k= 0
#     i=0
#     a = {}
#
#     # print(embeddings_index['how'])
#     oov = {}
#     for word in tqdm(vocab):
#         try:
#
#             # print( embeddings_index[word])
#             a[word] = embeddings_index[word]
#             k += vocab[word]
#         except:
#             oov[word] = vocab[word]
#             i += vocab[word]
#     print("embeddings found{} %".format(len(a)/len(vocab)))
#     print("not found {} %".format( i/(k+i) ))
#     # print(a)
#     # print(oov)
#
# counter(data['question_text'],embeddings_index)