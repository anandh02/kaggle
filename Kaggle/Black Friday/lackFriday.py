import numpy as np
import pandas as pd

data = pd.read_csv('BlackFriday.csv')


def prod(id):
    print(id)


def gender(g):
    if g == 'F':
        return 0
    else:
        return 1


def productId(id):
    ids = id[1:]
    return ids

data['Gender'] = data['Gender'].apply(gender)

data['Product_ID'] = data['Product_ID'].apply(productId)