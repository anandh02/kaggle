from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, MaxPooling2D, Conv2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time

Name = "Cat-and-Dog-cnn 64x2-{}".format(int(time.time()))

# tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))

pickle_in = open("X.pickle","rb")
X= pickle.load(pickle_in)

pickle_in = open("Y.pickle","rb")
y= pickle.load(pickle_in)

X = X/255.0

# gpu_options = tf.GPUOptions

dense_layers = [0,1,2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]


for dense_layer in dense_layers:
	for layer_size in layer_sizes:
		for conv_layer in conv_layers:
			Name = "{}-Conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))

			model = Sequential()

			model.add(Conv2D(layer_size, (3,3),input_shape= X.shape[1:]))
			model.add(Activation("relu"))
			model.add(MaxPooling2D(pool_size=(2,2)))

			for l in range(conv_layer-1):
				model.add(Conv2D(layer_size, (3,3),input_shape= X.shape[1:]))
				model.add(Activation("relu"))
				model.add(MaxPooling2D(pool_size=(2,2)))

			model.add(Flatten())

			for l in range(dense_layer):
				model.add(Dense(layer_size))
				model.add(Activation("relu"))

			model.add(Dense(1))
			model.add(Activation("sigmoid"))

			tensorboard = TensorBoard(log_dir="logs/{}".format(Name))
			# tensorboard --logdir=logs/

			model.compile(loss="binary_crossentropy",
			             optimizer="adam",
			             metrics=["accuracy"])

			model.fit(X, y,
					batch_size=32,
					validation_split=0.1,
					epochs = 10, 
					callbacks=[tensorboard])




