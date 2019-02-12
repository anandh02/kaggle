import pandas as pd
from ast import literal_eval

md = pd.read_csv('./data/movies_metadata.csv')

def gen(x):
    wor = []
    for i in x:
        if isinstance(x,list):
            wor.append(i['name'])
        else :
            []
    return wor
md['genres'].apply(gen)
md['genres'].apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])