'''
Model zoo - assortment of CNN architectures
'''

# from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils

from cnn_functions import load_training_data, euclidean_distance, eucl_dist_output_shape, tensorprod_softmax, sparse_Convolution2D, sparse_MaxPooling2D, TensorProd2D, set_weights, residual_block
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten, Lambda, merge
import cropping

import os
import datetime
import h5py

"""
Vanilla convnets
"""

def feature_net_31x31(n_channels = 1, n_features = 3, reg = 0.001, drop = 0.5, init = 'he_normal'):
	print "Using feature net 31x31"

	model = Sequential()
	model.add(Convolution2D(32, 4, 4, init = init, border_mode='valid', input_shape=(n_channels, 31, 31), W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(128, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(Convolution2D(200, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(200, init = init, W_regularizer = l2(reg)))
	model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))
	return model

def sparse_feature_net_31x31(batch_input_shape = (1,1,1080,1280), n_features = 3, drop = 0.5, reg = 0.001, init = 'he_normal', weights_path = None):

	model = Sequential()
	d = 1

	model.add(sparse_Convolution2D(32, 4, 4, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(128, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(200, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, init = init, W_regularizer = l2(reg)))
	model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	model = set_weights(model, weights_path)

	return model

def feature_net_61x61(n_features = 3, n_channels = 2, reg = 0.001, drop = 0.5, init = 'he_normal'):
	print "Using feature net 61x61"

	model = Sequential()
	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', input_shape=(n_channels, 61, 61), W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(200, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	if drop > 0:
		model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(200, init = init, W_regularizer = l2(reg)))
	if drop > 0:
		model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))

	return model

def sparse_feature_net_61x61(batch_input_shape = (1,2,1080,1280), n_features = 3, reg = 0.001, init = 'he_normal', weights_path = None):

	model = Sequential()
	d = 1
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(200, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, init = init, W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	model = set_weights(model, weights_path)

	return model

def feature_net_81x81(n_features = 3, n_channels = 2, reg = 1e-5, init = 'he_normal', drop = 0.5):
	print "Using feature net 81x81"
	model = Sequential()
	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', input_shape=(n_channels, 81, 81), W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(Convolution2D(200,4,4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	if drop > 0:
		model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(200, init = init, W_regularizer = l2(reg)))
	if drop > 0:
		model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))

	return model

def sparse_feature_net_81x81(batch_input_shape = (1,2,1080,1280), n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None):

	model = Sequential()

	d = 1
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	
	d *= 2
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	
	d *= 2
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))

	d *= 2
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(200, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, init = init, W_regularizer = l2(reg)))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	model = set_weights(model, weights_path)

	return model

def feature_net_101x101(n_features = 3, reg = 0.001, init = 'he_normal', drop = 0.5):
	print "Using feature net 101x101"
 
	model = Sequential()
	model.add(Convolution2D(32, 4, 4, init = init, border_mode='valid', input_shape=(2, 101, 101), W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(128, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(Convolution2D(128, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(Convolution2D(128, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(256, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(Convolution2D(256, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(Convolution2D(512, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	if drop > 0:
		model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(512, init = init, W_regularizer = l2(reg)))
	if drop > 0:
		model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))

	return model

def sparse_feature_net_101x101(batch_input_shape = (1,2,1080,1280), n_features = 3, reg = 0.001, init = 'he_normal', weights_path = None):

	d = 1
	model = Sequential()
	model.add(sparse_Convolution2D(32, 4, 4, d = d, init = init, border_mode='valid', input_shape=(2, 101, 101), W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))

	d *= 2
	model.add(sparse_Convolution2D(64, 4, 4, d = d,  init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_Convolution2D(64, 3, 3, d = d,  init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))

	d *= 2
	model.add(sparse_Convolution2D(128, 3, 3, d = d,  init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_Convolution2D(128, 3, 3, d = d,  init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_Convolution2D(128, 3, 3, d = d,  init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))

	d *= 2
	model.add(sparse_Convolution2D(256, 3, 3, d = d,  init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_Convolution2D(256, 3, 3, d = d,  init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(sparse_Convolution2D(512, 4, 4, d = d,  init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	if drop > 0:
		model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(TensorProd2D(512, 512, init = init, W_regularizer = l2(reg)))
	if drop > 0:
		model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(TensorProd2D(512, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	model = set_weights(model, weights_path)

	return model

"""
Batch normalized convnets
"""
def bn_feature_net_21x21(n_channels = 1, n_features = 3, reg = 1e-5, init = 'he_normal'):
	print "Using feature net 21x21 with batch normalization"

	model = Sequential()
	model.add(Convolution2D(32, 4, 4, init = init, border_mode='valid', input_shape=(n_channels, 21, 21), W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(32, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(32, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(32, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(200, 4, 4, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))
	return model

def bn_feature_net_31x31(n_channels = 1, n_features = 3, reg = 1e-5, init = 'he_normal'):
	print "Using feature net 31x31 with batch normalization"

	model = Sequential()
	model.add(Convolution2D(32, 4, 4, init = init, border_mode='valid', input_shape=(n_channels, 31, 31), W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(128, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(200, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))
	return model

def sparse_bn_feature_net_31x31(batch_input_shape = (1,1,1080,1280), n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None):

	model = Sequential()
	d = 1

	model.add(sparse_Convolution2D(32, 4, 4, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))	
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(128, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(200, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	model = set_weights(model, weights_path)

	return model

def bn_feature_net_61x61(n_features = 3, n_channels = 2, reg = 1e-5, init = 'he_normal'):
	print "Using feature net 61x61 with batch normalization"
	model = Sequential()
	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', input_shape=(n_channels, 61, 61), W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(200,4,4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))

	return model

def sparse_bn_feature_net_61x61(batch_input_shape = (1,2,1080,1280), n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None):

	model = Sequential()
	d = 1
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(200, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	model = set_weights(model, weights_path)

	return model

def bn_feature_net_81x81(n_features = 3, n_channels = 2, reg = 1e-5, init = 'he_normal'):
	print "Using feature net 81x81 with batch normalization"
	model = Sequential()
	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', input_shape=(n_channels, 81, 81), W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(200,4,4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))

	return model

def sparse_bn_feature_net_81x81(batch_input_shape = (1,2,1080,1280), n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None):

	model = Sequential()

	d = 1
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	
	d *= 2
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	
	d *= 2
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))

	d *= 2
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(200, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	model = set_weights(model, weights_path)

	return model

"""
Multiresolution networks
"""

def bn_feature_net_multires_31x31(n_features = 3, n_channels = 1, reg = 1e-5):
	print "Using multiresolution feature net 31x31 with batch normalization"

	inputs = Input(shape = (n_channels,31,31))
	layer1 = Convolution2D(32, 4, 4, init = 'he_normal', border_mode='valid', input_shape=(n_channels, 31, 31), W_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(layer1)
	act1 = Activation('relu')(norm1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(act1)

	layer2 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool1)
	norm2 = BatchNormalization(axis = 1)(layer2)
	act2 = Activation('relu')(norm2)

	layer3 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act2)
	norm3 = BatchNormalization(axis = 1)(layer3)
	act3 = Activation('relu')(norm3)
	pool2 = MaxPooling2D(pool_size=(2, 2))(act3)

	layer4 = Convolution2D(128, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	norm4 = BatchNormalization(axis = 1)(layer4)
	act4 = Activation('relu')(norm4)

	side_layer1 = Convolution2D(200,14,14, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	side_norm1 = BatchNormalization(axis = 1)(side_layer1)
	side_act1 = Activation('relu')(side_norm1)

	side_layer2 = Convolution2D(200,3,3, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(norm4)
	side_norm2 = BatchNormalization(axis = 1)(side_layer2)
	side_act2 = Activation('relu')(side_norm2)

	merge_layer1 = merge([side_act1, side_act2], mode = 'concat', concat_axis = 1)

	flat = Flatten()(merge_layer1)

	dense1 = Dense(512, init = 'he_normal', W_regularizer = l2(reg))(flat)
	dense_norm1 = BatchNormalization(axis = 1)(dense1)
	dense_act1 = Activation('relu')(dense_norm1)

	dense2 = Dense(n_features, init = 'he_normal', W_regularizer = l2(reg))(dense_act1)
	dense_act2 = Activation('softmax')(dense2)

	model = Model(input = inputs, output = dense_act2)

	return model

def sparse_bn_feature_net_multires_31x31(batch_input_shape = (1,2,1080,1280), n_features = 3, n_channels = 1, reg = 1e-5, weights_path = None):
	print "Using multiresolution feature net 31x31 with batch normalization"

	d = 1
	inputs = Input(shape = batch_input_shape[1:])
	layer1 = sparse_Convolution2D(32, 4, 4, d = d, init = 'he_normal', border_mode='valid', input_shape=(n_channels, 31, 31), W_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(layer1)
	act1 = Activation('relu')(norm1)
	pool1 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act1)
	
	d *= 2
	layer2 = sparse_Convolution2D(64, 3, 3, d =d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool1)
	norm2 = BatchNormalization(axis = 1)(layer2)
	act2 = Activation('relu')(norm2)

	layer3 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act2)
	norm3 = BatchNormalization(axis = 1)(layer3)
	act3 = Activation('relu')(norm3)
	pool2 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act3)

	d *= 2
	layer4 = sparse_Convolution2D(128, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	norm4 = BatchNormalization(axis = 1)(layer4)
	act4 = Activation('relu')(norm4)

	side_layer1 = sparse_Convolution2D(200, 14, 14, d = 2, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	side_norm1 = BatchNormalization(axis = 1)(side_layer1)
	side_act1 = Activation('relu')(side_norm1)

	side_layer2 = sparse_Convolution2D(200, 3, 3, d = 4, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(norm4)
	side_norm2 = BatchNormalization(axis = 1)(side_layer2)
	side_act2 = Activation('relu')(side_norm2)

	merge_layer1 = merge([side_act1, side_act2], mode = 'concat', concat_axis = 1)

	dense1 = TensorProd2D(400, 512, init = 'he_normal', W_regularizer = l2(reg))(merge_layer1)
	dense_norm1 = BatchNormalization(axis = 1)(dense1)
	dense_act1 = Activation('relu')(dense_norm1)

	dense2 = TensorProd2D(512, n_features, init = 'he_normal', W_regularizer = l2(reg))(dense_act1)
	dense_act2 = Activation(tensorprod_softmax)(dense2)

	model = Model(input = inputs, output = dense_act2)
	model = set_weights(model, weights_path)

	return model

def bn_feature_net_multires_61x61(n_features = 3, n_channels = 2, reg = 1e-5):
	print "Using multiresolution feature net 61x61 with batch normalization"

	inputs = Input(shape = (n_channels,61,61))
	layer1 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', input_shape=(n_channels, 61, 61), W_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(layer1)
	act1 = Activation('relu')(norm1)

	layer2 = Convolution2D(64, 4, 4, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(layer2)
	act2 = Activation('relu')(norm2)
	pool1 = MaxPooling2D(pool_size=(2, 2))(act2)

	layer3 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool1)
	norm3 = BatchNormalization(axis = 1)(layer3)
	act3 = Activation('relu')(norm3)

	layer4 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act3)
	norm4 = BatchNormalization(axis = 1)(layer4)
	act4 = Activation('relu')(norm4)
	pool2 = MaxPooling2D(pool_size=(2, 2))(act4)

	layer5 = Convolution2D(128, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	norm5 = BatchNormalization(axis = 1)(layer5)
	act5 = Activation('relu')(norm5)

	layer6 = Convolution2D(128, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act5)
	norm6 = BatchNormalization(axis = 1)(layer6)
	act6 = Activation('relu')(norm6)
	pool3 = MaxPooling2D(pool_size=(2, 2))(act6)

	side_layer1 = Convolution2D(256,28,28, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	side_norm1 = BatchNormalization(axis = 1)(side_layer1)
	side_act1 = Activation('relu')(side_norm1)

	side_layer2 = Convolution2D(256,12,12, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool2)
	side_norm2 = BatchNormalization(axis = 1)(side_layer2)
	side_act2 = Activation('relu')(side_norm2)

	side_layer3 = Convolution2D(256,4,4, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool3)
	side_norm3 = BatchNormalization(axis = 1)(side_layer3)
	side_act3 = Activation('relu')(side_norm3)

	merge_layer1 = merge([side_act1, side_act2, side_act3], mode = 'concat', concat_axis = 1)

	flat = Flatten()(merge_layer1)

	dense1 = Dense(512, init = 'he_normal', W_regularizer = l2(reg))(flat)
	dense_norm1 = BatchNormalization(axis = 1)(dense1)
	dense_act1 = Activation('relu')(dense_norm1)

	dense2 = Dense(n_features, init = 'he_normal', W_regularizer = l2(reg))(dense_act1)
	dense_act2 = Activation('softmax')(dense2)

	model = Model(input = inputs, output = dense_act2)

	return model


def feature_net_multires_61x61(n_features = 3, n_channels = 2, reg = 1e-5):
	print "Using multiresolution feature net 61x61 with batch normalization"

	inputs = Input(shape = (n_channels,61,61))
	layer1 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', input_shape=(n_channels, 61, 61), W_regularizer = l2(reg))(inputs)
	act1 = Activation('relu')(layer1)

	layer2 = Convolution2D(64, 4, 4, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act1)
	act2 = Activation('relu')(layer2)
	pool1 = MaxPooling2D(pool_size=(2, 2))(act2)

	layer3 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool1)
	act3 = Activation('relu')(layer3)

	layer4 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act3)
	act4 = Activation('relu')(layer4)
	pool2 = MaxPooling2D(pool_size=(2, 2))(act4)

	layer5 = Convolution2D(128, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	act5 = Activation('relu')(layer5)

	layer6 = Convolution2D(128, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act5)
	act6 = Activation('relu')(layer6)
	pool3 = MaxPooling2D(pool_size=(2, 2))(act6)

	side_layer1 = Convolution2D(256,28,28, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	side_act1 = Activation('relu')(side_layer1)

	side_layer2 = Convolution2D(256,12,12, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool2)
	side_act2 = Activation('relu')(side_layer2)

	side_layer3 = Convolution2D(256,4,4, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool3)
	side_act3 = Activation('relu')(side_layer3)

	merge_layer1 = merge([side_act1, side_act2, side_act3], mode = 'concat', concat_axis = 1)

	flat = Flatten()(merge_layer1)

	dense1 = Dense(512, init = 'he_normal', W_regularizer = l2(reg))(flat)
	dense_act1 = Activation('relu')(dense1)

	dense2 = Dense(n_features, init = 'he_normal', W_regularizer = l2(reg))(dense_act1)
	dense_act2 = Activation('softmax')(dense2)

	model = Model(input = inputs, output = dense_act2)

	return model

def sparse_feature_net_multires_61x61(batch_input_shape = (1,2,1080,1280), n_features = 3, n_channels = 2, reg = 1e-5, weights_path = None):
	print "Using multiresolution feature net 61x61"

	d = 1
	inputs = Input(shape = batch_input_shape[1:])
	layer1 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', batch_input_shape = batch_input_shape, W_regularizer = l2(reg))(inputs)
	act1 = Activation('relu')(layer1)

	layer2 = sparse_Convolution2D(64, 4, 4, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act1)
	act2 = Activation('relu')(layer2)
	pool1 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act2)
	d *= 2

	layer3 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool1)
	act3 = Activation('relu')(layer3)

	layer4 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act3)
	act4 = Activation('relu')(layer4)
	pool2 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act4)
	d *= 2

	layer5 = sparse_Convolution2D(128, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	act5 = Activation('relu')(layer5)

	layer6 = sparse_Convolution2D(128, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act5)
	act6 = Activation('relu')(layer6)
	pool3 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act6)
	d *= 2

	side_layer1 = sparse_Convolution2D(256,28,28, d = 2, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	side_act1 = Activation('relu')(side_layer1)

	side_layer2 = sparse_Convolution2D(256,12,12, d = 4, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool2)
	side_act2 = Activation('relu')(side_layer2)

	side_layer3 = sparse_Convolution2D(256,4,4, d = 8, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool3)
	side_act3 = Activation('relu')(side_layer3)

	merge_layer1 = merge([side_act1, side_act2, side_act3], mode = 'concat', concat_axis = 1)

	dense1 = TensorProd2D(768, 512, init = 'he_normal', W_regularizer = l2(reg))(merge_layer1)
	dense_act1 = Activation('relu')(dense1)

	dense2 = TensorProd2D(512, n_features, init = 'he_normal', W_regularizer = l2(reg))(dense_act1)
	dense_act2 = Activation(tensorprod_softmax)(dense2)

	model = Model(input = inputs, output = dense_act2)
	model = set_weights(model, weights_path)

	return model

def sparse_bn_feature_net_multires_61x61(batch_input_shape = (1,2,1080,1280), n_features = 3, n_channels = 2, reg = 1e-5, weights_path = None):
	print "Using multiresolution feature net 61x61 with batch normalization"

	d = 1
	inputs = Input(shape = batch_input_shape[1:])
	layer1 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', batch_input_shape = batch_input_shape, W_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(layer1)
	act1 = Activation('relu')(norm1)

	layer2 = sparse_Convolution2D(64, 4, 4, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(layer2)
	act2 = Activation('relu')(norm2)
	pool1 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act2)
	d *= 2

	layer3 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool1)
	norm3 = BatchNormalization(axis = 1)(layer3)
	act3 = Activation('relu')(norm3)

	layer4 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act3)
	norm4 = BatchNormalization(axis = 1)(layer4)
	act4 = Activation('relu')(norm4)
	pool2 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act4)
	d *= 2

	layer5 = sparse_Convolution2D(128, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	norm5 = BatchNormalization(axis = 1)(layer5)
	act5 = Activation('relu')(norm5)

	layer6 = sparse_Convolution2D(128, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act5)
	norm6 = BatchNormalization(axis = 1)(layer6)
	act6 = Activation('relu')(norm6)
	pool3 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act6)
	d *= 2

	side_layer1 = sparse_Convolution2D(256,28,28, d = 2, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	side_norm1 = BatchNormalization(axis = 1)(side_layer1)
	side_act1 = Activation('relu')(side_norm1)

	side_layer2 = sparse_Convolution2D(256,12,12, d = 4, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool2)
	side_norm2 = BatchNormalization(axis = 1)(side_layer2)
	side_act2 = Activation('relu')(side_norm2)

	side_layer3 = sparse_Convolution2D(256,4,4, d = 8, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool3)
	side_norm3 = BatchNormalization(axis = 1)(side_layer3)
	side_act3 = Activation('relu')(side_norm3)

	merge_layer1 = merge([side_act1, side_act2, side_act3], mode = 'concat', concat_axis = 1)

	dense1 = TensorProd2D(768, 512, init = 'he_normal', W_regularizer = l2(reg))(merge_layer1)
	dense_norm1 = BatchNormalization(axis = 1)(dense1)
	dense_act1 = Activation('relu')(dense_norm1)

	dense2 = TensorProd2D(512, n_features, init = 'he_normal', W_regularizer = l2(reg))(dense_act1)
	dense_act2 = Activation(tensorprod_softmax)(dense2)

	model = Model(input = inputs, output = dense_act2)
	model = set_weights(model, weights_path)

	return model

def bn_feature_net_multires_81x81(n_features = 3, n_channels = 2, reg = 1e-5):
	print "Using multiresolution feature net 81x81 with batch normalization"

	inputs = Input(shape = (n_channels,81,81))
	layer1 = Convolution2D(32, 3, 3, init = 'he_normal', border_mode='valid', input_shape=(n_channels, 81, 81), W_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(layer1)
	act1 = Activation('relu')(norm1)

	layer2 = Convolution2D(32, 4, 4, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(layer2)
	act2 = Activation('relu')(norm2)
	pool1 = MaxPooling2D(pool_size=(2, 2))(act2)

	layer3 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool1)
	norm3 = BatchNormalization(axis = 1)(layer3)
	act3 = Activation('relu')(norm3)

	layer4 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act3)
	norm4 = BatchNormalization(axis = 1)(layer4)
	act4 = Activation('relu')(norm4)

	layer5 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act4)
	norm5 = BatchNormalization(axis = 1)(layer5)
	act5 = Activation('relu')(norm5)
	pool2 = MaxPooling2D(pool_size=(2, 2))(act5)

	layer6 = Convolution2D(128, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	norm6 = BatchNormalization(axis = 1)(layer6)
	act6 = Activation('relu')(norm6)

	layer7 = Convolution2D(128, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act6)
	norm7 = BatchNormalization(axis = 1)(layer7)
	act7 = Activation('relu')(norm7)
	pool3 = MaxPooling2D(pool_size=(2, 2))(act7)

	layer8 = Convolution2D(128, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool3)
	norm8 = BatchNormalization(axis = 1)(layer8)
	act8 = Activation('relu')(norm8)

	
	side_layer1 = Convolution2D(256,32,32, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act5)
	side_norm1 = BatchNormalization(axis = 1)(side_layer1)
	side_act1 = Activation('relu')(side_norm1)

	side_layer2 = Convolution2D(256,12,12, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act7)
	side_norm2 = BatchNormalization(axis = 1)(side_layer2)
	side_act2 = Activation('relu')(side_norm2)

	side_layer3 = Convolution2D(256,4,4, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act8)
	side_norm3 = BatchNormalization(axis = 1)(side_layer3)
	side_act3 = Activation('relu')(side_norm3)

	merge_layer1 = merge([side_act1, side_act2, side_act3], mode = 'concat', concat_axis = 1)

	flat = Flatten()(merge_layer1)

	dense1 = Dense(512, init = 'he_normal', W_regularizer = l2(reg))(flat)
	dense_norm1 = BatchNormalization(axis = 1)(dense1)
	dense_act1 = Activation('relu')(dense_norm1)

	dense2 = Dense(n_features, init = 'he_normal', W_regularizer = l2(reg))(dense_act1)
	dense_act2 = Activation('softmax')(dense2)

	model = Model(input = inputs, output = dense_act2)

	return model

def bn_sparse_feature_net_multires_81x81(batch_input_shape = (1,2,1080,1280), n_features = 3, n_channels = 2, reg = 1e-5, weights_path = None):
	print "Using multiresolution feature net 81x81 with batch normalization"

	inputs = Input(shape = batch_input_shape[1:])
	d = 1
	layer1 = sparse_Convolution2D(32, 3, 3, d = d, init = 'he_normal', border_mode='valid', input_shape=(n_channels, 81, 81), W_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(layer1)
	act1 = Activation('relu')(norm1)

	layer2 = sparse_Convolution2D(32, 4, 4, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(layer2)
	act2 = Activation('relu')(norm2)
	pool1 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act2)
	d *= 2

	layer3 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool1)
	norm3 = BatchNormalization(axis = 1)(layer3)
	act3 = Activation('relu')(norm3)

	layer4 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act3)
	norm4 = BatchNormalization(axis = 1)(layer4)
	act4 = Activation('relu')(norm4)

	layer5 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act4)
	norm5 = BatchNormalization(axis = 1)(layer5)
	act5 = Activation('relu')(norm5)
	pool2 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act5)
	d*= 2

	layer6 = sparse_Convolution2D(128, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	norm6 = BatchNormalization(axis = 1)(layer6)
	act6 = Activation('relu')(norm6)

	layer7 = sparse_Convolution2D(128, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act6)
	norm7 = BatchNormalization(axis = 1)(layer7)
	act7 = Activation('relu')(norm7)
	pool3 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act7)
	d *= 2

	layer8 = sparse_Convolution2D(128, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool3)
	norm8 = BatchNormalization(axis = 1)(layer8)
	act8 = Activation('relu')(norm8)

	
	side_layer1 = sparse_Convolution2D(256,32,32,d = 2, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act5)
	side_norm1 = BatchNormalization(axis = 1)(side_layer1)
	side_act1 = Activation('relu')(side_norm1)

	side_layer2 = sparse_Convolution2D(256,12,12, d = 4, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act7)
	side_norm2 = BatchNormalization(axis = 1)(side_layer2)
	side_act2 = Activation('relu')(side_norm2)

	side_layer3 = sparse_Convolution2D(256,4,4, d = 8, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act8)
	side_norm3 = BatchNormalization(axis = 1)(side_layer3)
	side_act3 = Activation('relu')(side_norm3)

	merge_layer1 = merge([side_act1, side_act2, side_act3], mode = 'concat', concat_axis = 1)

	dense1 = TensorProd2D(768, 512, init = 'he_normal', W_regularizer = l2(reg))(merge_layer1)
	dense_norm1 = BatchNormalization(axis = 1)(dense1)
	dense_act1 = Activation('relu')(dense_norm1)

	dense2 = TensorProd2D(512, n_features, init = 'he_normal', W_regularizer = l2(reg))(dense_act1)
	dense_act2 = Activation(tensorprod_softmax)(dense2)

	model = Model(input = inputs, output = dense_act2)
	model = set_weights(model, weights_path)

	return model

def feature_net_multires_101x101(n_features = 3, reg = 0.001, drop = 0.5):

	inputs = Input(shape = (2,101,101))
	layer1 = Convolution2D(32, 4, 4, name = 'layer1', init = 'he_normal', border_mode='valid', input_shape=(2, 101, 101), W_regularizer = l2(reg))(inputs)
	act1 = Activation('relu')(layer1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(act1)

	layer2 = Convolution2D(64, 4, 4, name = 'layer2', init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	act2 = Activation('relu')(layer2)
	layer3 = Convolution2D(64, 3, 3, name = 'layer3', init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act2)
	act3 = Activation('relu')(layer3)
	pool2 = MaxPooling2D(pool_size=(2, 2))(act3)

	layer4 = Convolution2D(128, 3, 3, name = 'layer4', init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	act4 = Activation('relu')(layer4)
	layer5 = Convolution2D(128, 3, 3, name = 'layer5', init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act4)
	act5 = Activation('relu')(layer5)
	layer6 = Convolution2D(128, 3, 3, name = 'layer6', init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act5)
	act6 = Activation('relu')(layer6)
	pool3 = MaxPooling2D(pool_size=(2, 2))(act6)

	side_layer1 = Convolution2D(512,16,16, name = 'layer9', init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(layer6)
	dropside1 = Dropout(drop)(side_layer1)

	layer7 = Convolution2D(256, 3, 3, name = 'layer7', init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool3)
	act7 = Activation('relu')(layer7)
	layer8 = Convolution2D(256, 3, 3, name = 'layer8', init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act7)
	act8 = Activation('relu')(layer8)
	layer9 = Convolution2D(512,4,4, name = 'layer9', init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act8)
	drop1 = Dropout(drop)(layer9)
	act9 = Activation('relu')(drop1)

	merge_layer1 = merge([side_layer1, layer9], mode = 'concat', concat_axis = 1)

	flat = Flatten()(merge_layer1)

	dense1 = Dense(1024, name = 'layer10', init = 'he_normal', W_regularizer = l2(reg))(flat)
	drop2 = Dropout(drop)(dense1)
	act10 = Activation('relu')(drop2)
	dense2 = Dense(n_features, name = 'layer11', init = 'he_normal', W_regularizer = l2(reg))(act10)
	act11 = Activation('softmax')(dense2)

	model = Model(input = inputs, output = act11)

	return model

def sparse_feature_net_multires_101x101(batch_input_shape = (1,2,1080,1280), n_features = 3, reg = 0.001, weights_path = None):

	d = 1
	inputs = Input(shape = batch_input_shape[1:])
	layer1 = sparse_Convolution2D(32, 4, 4, name = 'layer1', d = d, init = 'he_normal', border_mode='valid', input_shape=(2, 101, 101), W_regularizer = l2(reg))(inputs)
	act1 = Activation('relu')(layer1)
	pool1 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act1)

	d *= 2
	layer2 = sparse_Convolution2D(64, 4, 4, name = 'layer2', d = d,  init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	act2 = Activation('relu')(layer2)
	layer3 = sparse_Convolution2D(64, 3, 3, name = 'layer3', d = d,  init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act2)
	act3 = Activation('relu')(layer3)
	pool2 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act3)

	d*= 2
	layer4 = sparse_Convolution2D(128, 3, 3, name = 'layer4', d = d,  init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	act4 = Activation('relu')(layer4)
	layer5 = sparse_Convolution2D(128, 3, 3, name = 'layer5', d = d,  init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act4)
	act5 = Activation('relu')(layer5)
	layer6 = sparse_Convolution2D(128, 3, 3, name = 'layer6', d = d,  init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act5)
	act6 = Activation('relu')(layer6)
	pool3 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act6)

	side_layer1 = sparse_Convolution2D(512,16,16, name = 'layer9', d = d,  init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(layer6)
	dropside1 = Dropout(drop)(side_layer1)

	d *= 2
	layer7 = sparse_Convolution2D(256, 3, 3, name = 'layer7', d = d,  init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool3)
	act7 = Activation('relu')(layer7)
	layer8 = sparse_Convolution2D(256, 3, 3, name = 'layer8', d = d,  init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act7)
	act8 = Activation('relu')(layer8)
	layer9 = sparse_Convolution2D(512,4,4, name = 'layer9', d = d,  init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act8)
	drop1 = Dropout(drop)(layer9)
	act9 = Activation('relu')(drop1)

	merge_layer1 = merge([dropside1, act9], mode = 'concat')

	dense1 = TensorProd2D(1024, 1024, name = 'layer10', init = 'he_normal', W_regularizer = l2(reg))(merge_layer1)
	drop2 = Dropout(drop)(dense1)
	act10 = Activation('relu')(drop2)
	dense2 = TensorProd2D(1024, n_features, name = 'layer11', init = 'he_normal', W_regularizer = l2(reg))(act10)
	act11 = Activation(tensorprod_softmax)(dense2)

	model = Model(input = inputs, output = act11)

	model = set_weights(model, weights_path)

	return model

def bn_feature_net_multires_101x101(n_channels = 2, n_features = 3, reg = 0.001, drop = 0.5):

	inputs = Input(shape = (n_channels,101,101))
	layer1 = Convolution2D(64, 4, 4, init = 'he_normal', border_mode='valid', input_shape=(2, 101, 101), W_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(layer1)
	act1 = Activation('relu')(norm1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(act1)

	layer2 = Convolution2D(64, 4, 4, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	norm2 = BatchNormalization(axis = 1)(layer2)
	act2 = Activation('relu')(norm2)
	layer3 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act2)
	norm3 = BatchNormalization(axis = 1)(layer3)
	act3 = Activation('relu')(norm3)
	pool2 = MaxPooling2D(pool_size=(2, 2))(act3)

	layer4 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	norm4 = BatchNormalization(axis = 1)(layer4)
	act4 = Activation('relu')(norm4)
	layer5 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act4)
	norm5 = BatchNormalization(axis = 1)(layer5)
	act5 = Activation('relu')(norm5)
	layer6 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act5)
	norm6 = BatchNormalization(axis = 1)(layer6)
	act6 = Activation('relu')(norm6)
	pool3 = MaxPooling2D(pool_size=(2, 2))(act6)

	side_layer1 = Convolution2D(200,16,16, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(layer6)
	norm_side_layer1 = BatchNormalization(axis = 1)(side_layer1)
	act_side_layer1 = Activation('relu')(norm_side_layer1)

	layer7 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool3)
	norm7 = BatchNormalization(axis = 1)(layer7)
	act7 = Activation('relu')(norm7)
	layer8 = Convolution2D(64, 3, 3, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act7)
	norm8 = BatchNormalization(axis = 1)(layer8)
	act8 = Activation('relu')(norm8)
	layer9 = Convolution2D(200,4,4, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act8)
	norm9 = BatchNormalization(axis = 1)(layer9)
	act9 = Activation('relu')(norm9)

	merge_layer1 = merge([act_side_layer1, act9], mode = 'concat', concat_axis = 1)

	flat = Flatten()(merge_layer1)

	dense1 = Dense(200, init = 'he_normal', W_regularizer = l2(reg))(flat)
	norm_dense1 = BatchNormalization(axis = 1)(dense1)
	act10 = Activation('relu')(norm_dense1)
	dense2 = Dense(n_features, init = 'he_normal', W_regularizer = l2(reg))(act10)
	act11 = Activation('softmax')(dense2)

	model = Model(input = inputs, output = act11)

	return model

def sparse_bn_feature_net_multires_101x101(batch_input_shape = (1,2,1080,1280), n_features = 3, reg = 0.001, weights_path = None):

	d = 1
	inputs = Input(shape = batch_input_shape[1:])
	layer1 = sparse_Convolution2D(64, 4, 4, d = d, init = 'he_normal', border_mode='valid', input_shape=(2, 101, 101), W_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(layer1)
	act1 = Activation('relu')(norm1)
	pool1 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act1)
	d *= 2

	layer2 = sparse_Convolution2D(64, 4, 4, d = d, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	norm2 = BatchNormalization(axis = 1)(layer2)
	act2 = Activation('relu')(norm2)
	layer3 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act2)
	norm3 = BatchNormalization(axis = 1)(layer3)
	act3 = Activation('relu')(norm3)
	pool2 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act3)
	d *= 2

	layer4 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool2)
	norm4 = BatchNormalization(axis = 1)(layer4)
	act4 = Activation('relu')(norm4)
	layer5 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act4)
	norm5 = BatchNormalization(axis = 1)(layer5)
	act5 = Activation('relu')(norm5)
	layer6 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(act5)
	norm6 = BatchNormalization(axis = 1)(layer6)
	act6 = Activation('relu')(norm6)
	pool3 = sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))(act6)
	d *= 2

	side_layer1 = sparse_Convolution2D(200,16,16, d = 4, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(layer6)
	norm_side_layer1 = BatchNormalization(axis = 1)(side_layer1)
	act_side_layer1 = Activation('relu')(norm_side_layer1)

	layer7 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode='valid', W_regularizer = l2(reg))(pool3)
	norm7 = BatchNormalization(axis = 1)(layer7)
	act7 = Activation('relu')(norm7)
	layer8 = sparse_Convolution2D(64, 3, 3, d = d, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act7)
	norm8 = BatchNormalization(axis = 1)(layer8)
	act8 = Activation('relu')(norm8)
	layer9 = sparse_Convolution2D(200, 4, 4, d = d, init = 'he_normal', border_mode = 'valid', W_regularizer = l2(reg))(act8)
	norm9 = BatchNormalization(axis = 1)(layer9)
	act9 = Activation('relu')(norm9)

	merge_layer1 = merge([act_side_layer1, act9], mode = 'concat', concat_axis = 1)

	dense1 = TensorProd2D(400, 200, init = 'he_normal', W_regularizer = l2(reg))(merge_layer1)
	norm_dense1 = BatchNormalization(axis = 1)(dense1)
	act10 = Activation('relu')(norm_dense1)
	dense2 = TensorProd2D(200, n_features, init = 'he_normal', W_regularizer = l2(reg))(act10)
	act11 = Activation(tensorprod_softmax)(dense2)

	model = Model(input = inputs, output = act11)
	model = set_weights(model, weights_path)

	for layer in model.layers:
		print layer.name, layer.output_shape
	return model

"""
Siamese networks
"""
def siamese_net_51x51(reg = 0.001, drop = 0.5, init = 'he_normal'):
	seq = Sequential()
	seq.add(Convolution2D(32, 4, 4, name = 'layer1', init = init, border_mode='valid', input_shape=(1, 51, 51), W_regularizer = l2(reg)))
	seq.add(Activation('relu'))
	seq.add(MaxPooling2D(pool_size=(2, 2)))

	seq.add(Convolution2D(64, 3, 3, name = 'layer2', init = init, border_mode='valid', W_regularizer = l2(reg)))
	seq.add(Convolution2D(64, 3, 3, name = 'layer3', init = init, border_mode='valid', W_regularizer = l2(reg)))
	seq.add(MaxPooling2D(pool_size=(2, 2)))

	seq.add(Convolution2D(64, 3, 3, name = 'layer4', init = init, border_mode='valid', W_regularizer = l2(reg)))
	seq.add(MaxPooling2D(pool_size=(2, 2)))

	seq.add(Flatten())
	seq.add(Dense(256, name = 'layer5', init = init, W_regularizer = l2(reg)))
	if drop > 0:
		seq.add(Dropout(drop))

	seq.add(Dense(256, name = 'layer6', init = init, W_regularizer = l2(reg)))
	if drop > 0:
		seq.add(Dropout(drop))

	input_1 = Input(shape = (1,51,51), name = 'input_1')
	input_2 = Input(shape = (1,51,51), name = 'input_2')

	processed_1 = seq(input_1)
	processed_2 = seq(input_2)

	distance = Lambda(euclidean_distance, output_shape = eucl_dist_output_shape)([processed_1, processed_2])
	model = Model(input = [input_1, input_2], output = distance)

	return model

def simple_siamese(reg = 0.001, drop = 0.5, init = 'he_normal'):
	
	seq = Sequential()
	seq.add(Dense(128, input_shape=(51*51,), activation='relu'))
	seq.add(Dropout(0.1))
	seq.add(Dense(128, activation='relu'))
	seq.add(Dropout(0.1))
	seq.add(Dense(128, activation='relu'))

	input_1 = Input(shape = (1,51,51), name = 'input_1')
	input_2 = Input(shape = (1,51,51), name = 'input_2')

	flat_1 = Flatten()(input_1)
	flat_2 = Flatten()(input_2)

	processed_1 = seq(flat_1)
	processed_2 = seq(flat_2)

	distance = Lambda(euclidean_distance, output_shape = eucl_dist_output_shape)([processed_1, processed_2])
	model = Model(input = [input_1, input_2], output = distance)

	return model

''' Residual network block designs '''
def residual_unit_1L(n_filters, kernel, init = 'he_normal', reg = 0.001):
	def f(input):
		norm1 = BatchNormalization(axis = 1, mode = 2)(input)
		act1 = Activation('relu')(norm1)
		conv1 = Convolution2D(n_filters, kernel, kernel, init=init, border_mode = 'valid', W_regularizer = l2(reg))(act1)

		crop_size = (kernel - 1)/2
		#short1 = Convolution2D(n_filters, 3, 3, init=init, border_mode = 'valid', W_regularizer = l2(reg))(input)
		short1 = cropping.Cropping2D(cropping=((crop_size,crop_size),(crop_size,crop_size)) )(input)
		return merge([short1, conv1], mode="sum")

	return f

'''Residual network architectures '''
#residual_block is a helper function in cnn_function.py
def resnet_61x61(n_channels, n_categories):
	input = Input(shape=(n_channels,61,61))
	#(1,61,61)
	block1 = residual_block(residual_unit_1L, 64, 3, 1)(input)
	#(64, 59, 59)
	block2 = residual_block(residual_unit_1L, 64, 4, 1)(block1)
	#(64, 56, 56)
	pool1 = MaxPooling2D(pool_size=(2,2))(block2)
	#(64, 28, 28)
	block3 = residual_block(residual_unit_1L, 128, 3, 1)(pool1)
	#(128, 26, 26)
	pool2 = MaxPooling2D(pool_size = (2,2))(block3)
	#(128, 13, 13)
	block4 = residual_block(residual_unit_1L, 256, 4, 1)(pool2)
	#(256, 10, 10)
	pool3 = MaxPooling2D(pool_size = (2,2))(block4)
	#(256, 5, 5)

	flatten1 = Flatten()(pool3)
	dense1 = Dense(output_dim=n_categories, init = init, activation = "softmax", W_regularizer = l2(reg))(flatten1)
	model = Model(input = input, output = dense1)

	return model
