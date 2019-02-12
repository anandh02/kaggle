import pandas as pd
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout, Input, BatchNormalization, Activation
from keras.optimizers import Nadam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
# import numpy as npcannot import name 'check_numerics'


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train.drop(['Id','groupId','matchId','matchType'],axis =1,inplace=True)
test.drop(['Id','groupId','matchId','matchType'],axis =1,inplace=True)

train = train.sample(frac=0.08)
test =test.sample(frac=0.08)

Xtrain = train.iloc[:,0:24]
Ytrain= train.iloc[:,24:]

# Xtrain, XVtrain, Ytrain, YVtrain = train_test_split(Xtrain,Ytrain, test_size=0.1)

Xtest = test.iloc[:,0:24]
Ytest= test.iloc[:,24:]

model = Sequential()
model.add(Dense(24,input_dim = 24 ))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('relu'))
model.summary()
model.compile(optimizer='sgd',
              loss='mean_squared_error',
              # optimizer='adam'
             )
model.fit(Xtrain, Ytrain,batch_size=10)
#, validation_data = (XVtrain,YVtrain))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, Ytrain,batch_size=10)
scores = model.evaluate(Xtrain, Ytrain)

model.predict(Xtrain)

