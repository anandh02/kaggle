import pandas as pd
# import pixiedust

data = pd.read_csv('./Sentiment.csv')


data = data[['sentiment','text']]


Xtrain_pos = data[data['sentiment'] == 'Positive']

Xtrain_neg = data[data['sentiment'] == 'Negative']

# %%cython
def cleaning(data,color = 'white'):
    words = ' '.join(data)
    # print(words)
    # wor= []
    # for word in words.split():
    #     if 'http' not in word and not word.startswith('@') and not word.startswith('#') and word != 'RT' :
    #         wor.append(word)
    #
    # print(wor)

    # word = " ".join([word for word in words.split()
    #                 if 'http' not in word
    #                 and not word.startswith('@')
    #                 and not word.startswith('.@')
    #                 and not word.startswith('#')
    #                 and word != 'RT'])
    # print(word)
    # return word


wor = cleaning(Xtrain_pos['text'])