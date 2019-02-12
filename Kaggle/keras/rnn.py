import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM,Activation, CuDNNLSTM

mnist = tf.keras.datasets.mnist
(xtrain,ytrain),(xtest,ytest) = mnist.load_data()

xtrain = xtrain/255.0
xtest = xtest/255.0

model = Sequential()
# model.add(LSTM(128,input_shape = xtrain.shape[1:], activation ='relu', return_sequences = True))
# model.add(Dropout(0.2))

# model.add(LSTM(128, activation ='relu', return_sequences = False))
# model.add(Dropout(0.2))


model.add(CuDNNLSTM(128,input_shape = xtrain.shape[1:], return_sequences = True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128, return_sequences = False))
model.add(Dropout(0.2))


model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(10,activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss="sparse_categorical_crossentropy", 
			optimizer=opt,
			metrics=['accuracy'])

model.fit(xtrain,ytrain,
		epochs=3,
		validation_data=(xtest,ytest))

print(xtrain.shape)
print(ytrain.shape) 