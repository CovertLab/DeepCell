from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, merge, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
import h5py

reg = 0.001
drop = 0.5
init = 'he_normal'
n_channels = 1
x_pixel = 61
y_pixel = 61

#Defines block creating function
def residual_block(block_function, n_filters, reps):
	def f(input):
		for i in range(reps):
			input = block_function(n_filters = n_filters)(input)
		return input

	return f

''' Define different resnet unit blocks '''
def residual_unit_3L(n_filters):
	def f(input):
		norm1 = BatchNormalization(axis = 1)(input)
		act1 = Activation('relu')(norm1)
		conv1 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act1)
		
		norm2 = BatchNormalization(axis = 1)(conv1)
		act2 = Activation('relu')(norm2)
		conv2 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act2)

		norm3 = BatchNormalization(axis = 1)(conv2)
		act3 = Activation('relu')(norm3)
		conv3 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act3)
		
		return merge([input, conv3], mode = "sum")

	return f

def residual_unit_2L(n_filters):
	def f(input):
		norm1 = BatchNormalization(axis = 1)(input)
		act1 = Activation('relu')(norm1)
		conv1 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act1)
		
		norm2 = BatchNormalization(axis = 1)(conv1)
		act2 = Activation('relu')(norm2)
		conv2 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act2)

		return merge([input, conv2], mode = "sum")

	return f

def bottleneck_unit(n_filters):
	#follows the design from http://arxiv.org/pdf/1512.03385v1.pdf
	def f(input):
		norm1 = BatchNormalization(axis = 1)(input)
		act1 = Activation('relu')(norm1)
		conv1 = Convolution2D(n_filters, 1, 1, init=init, border_mode='same', W_regularizer = l2(reg))(act1)

		norm2 = BatchNormalization(axis = 1)(conv1)
		act2 = Activation('relu')(norm2)
		conv2 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act2)

		norm3 = BatchNormalization(axis = 1)(conv2)
		act3 = Activation('relu')(norm3)
		conv3 = Convolution2D(n_filters*4, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act3)

		#need to convolve the input to change depth shape for merge to be valid
		short1 = Convolution2D(n_filters*4, 1, 1, init=init, border_mode='same', W_regularizer = l2(reg))(input)

		return merge([short1, conv3])

	return f

''' Define different resnet architectures '''
def resnet_61x61(n_channels, n_categories, n_unit1 = 1, n_unit2 = 1, n_unit3 = 1):
	input = Input(shape=(n_channels,61,61))

	conv1 = Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg))(input)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)
	#now the shape = (64, 59, 59)
	conv2 = Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	#now shape = (64, 56, 56)
	block1 = residual_block(residual_unit_2L, 64, n_unit1)(act2)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	#now shape = (64, 28, 28)
	conv3 = Convolution2D(128, 1, 1, init = init, border_mode = 'same', W_regularizer = l2(reg))(pool1)
	#now shape = (128, 28, 28)
	block2 = residual_block(residual_unit_2L, 128, n_unit2)(conv3)
	pool2 = MaxPooling2D(pool_size=(2,2))(block2)
	#now shape = (128, 14, 14)
	conv4 = Convolution2D(256, 1, 1, init = init, border_mode = 'same', W_regularizer = l2(reg))(pool2)
	#now shape = (256, 14, 14)
	block3 = residual_block(residual_unit_2L, 256, n_unit3)(conv4)
	pool3 = MaxPooling2D(pool_size=(2,2))(block3)
	#now shape = (64, 7, 7)

	flatten1 = Flatten()(pool3)
	dense1 = Dense(output_dim=200, init = init, activation = "relu", W_regularizer = l2(reg))(flatten1)
	dense2 = Dense(output_dim=n_categories, init = init, activation = "softmax", W_regularizer = l2(reg))(dense1)

	model = Model(input = input, output = dense2)

	return model

def bottle_net_61x61(n_channels, n_categories, n_unit1 = 1, n_unit2 = 1, n_unit3 = 1):
	#inspired by http://arxiv.org/pdf/1512.03385v1.pdf
	input = Input(shape=(n_channels,61,61))

	conv1 = Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg))(input)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)
	#now the shape = (64, 59, 59)
	conv2 = Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	#now shape = (64, 56, 56)
	block1 = residual_block(bottleneck_unit, 64, n_unit1)(act2)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	#now shape = (64, 28, 28)
	block2 = residual_block(bottleneck_unit, 128, n_unit2)(pool1)  #(conv3)
	pool2 = MaxPooling2D(pool_size=(2,2))(block2)
	#now shape = (128, 14, 14)
	block3 = residual_block(bottleneck_unit, 256, n_unit3)(pool2)  #(conv4)
	pool3 = MaxPooling2D(pool_size=(2,2))(block3)
	#now shape = (64, 7, 7)

	flatten1 = Flatten()(pool3)
	dense1 = Dense(output_dim=200, init = init, activation = "relu", W_regularizer = l2(reg))(flatten1)
	dense2 = Dense(output_dim=n_categories, init = init, activation = "softmax", W_regularizer = l2(reg))(dense1)

	model = Model(input = input, output = dense2)

	return model





