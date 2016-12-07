import numpy as np
import tensorflow as tf

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Convolution1D, MaxPooling1D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils
from keras.models import model_from_json
import os

class cnn_model:
	def __init__(self):
		self.model = Sequential()


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)
cnn_m = cnn_model()

# 1. train
######################################
# TODO implement your training here!
# you can get the full data from the transition table like this:
#
# # both train_data and valid_data contain tupes of images and labels
# train_data = trans.get_train()
# valid_data = trans.get_valid()
# 
# alternatively you can get one random mini batch line this
#
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
######################################

# Get train and validation data
train_data = trans.get_train()
valid_data = trans.get_valid()

x = train_data[0].copy()
y = train_data[1].copy()

x_val = valid_data[0].copy()
y_val = valid_data[1].copy()

y = y.astype(int)
x = x.reshape(x.shape[0], 1, x.shape[1], 1)
print(y_val)
print("xxxxxxxxxxx")
y_val = y_val.astype(int)
print(y_val)
x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1], 1)

# Initialize keras model
# model = Sequential()

# Create cnn
print(train_data[0].shape)
cnn_m.model.add(Convolution2D(64, 3, 1, border_mode='same', input_shape=(1, 2500, 1)))
cnn_m.model.add(Activation('relu'))
cnn_m.model.add(Convolution2D(32, 3, 1, border_mode='same'))
cnn_m.model.add(Activation('relu'))
cnn_m.model.add(MaxPooling2D(pool_size=(1, 2)))
cnn_m.model.add(Dropout(0.25))

# cnn_m.model.add(Convolution1D(64, 3, border_mode='same'))
# cnn_m.model.add(Activation('relu'))
# cnn_m.model.add(Convolution1D(64, 3))
# cnn_m.model.add(Activation('relu'))
# cnn_m.model.add(MaxPooling1D(pool_length=(2)))
# cnn_m.model.add(Dropout(0.25))

cnn_m.model.add(Flatten())
cnn_m.model.add(Dense(512))
cnn_m.model.add(Activation('relu'))
cnn_m.model.add(Dropout(0.5))
cnn_m.model.add(Dense(5))
cnn_m.model.add(Activation('softmax'))


"""
self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = train_data[0].shape))
print("print train_data[0].shape ", train_data[0].shape)
#self.model.add(Dense(output_dim=64, input_dim=train_data.shape))
self.model.add(Activation("relu"))
self.model.add(Activation("softmax"))
"""

# Define attributes of the cnn; categorial, optimizer_type, performance metrics
cnn_m.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Define values for SGD optimizer
# from keras.optimizers import SGD
# cnn_m.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

# Fit the model to the training data
cnn_m.model.fit(x, y, nb_epoch=1, batch_size=64, show_accuracy=True, validation_data=(x_val, y_val), shuffle=True)

# Compute loss metrics for the current model i.e, training error
# cnn_m.model.loss_and_metrics = cnn_m.model.evaluate(valid_data[0], valid_data[1], batch_size=32)
# print("Valid error: ", cnn_m.model.loss_and_metrics)

# serialize model to JSON
model_json = cnn_m.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn_m.model.save_weights("model.h5")
print("Saved model to disk")





# 2. save your trained model


