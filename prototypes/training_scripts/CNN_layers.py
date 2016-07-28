"""
CNN layers - Classes for layers for convolutional neural networks
Builds upon the Keras layer
"""

"""
Import python packages
"""
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import shelve
from contextlib import closing

import os
import glob
import re
import numpy as np
import tifffile as tiff
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread

from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.tensor.signal.downsample import max_pool_2d as max_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse

from keras import backend as K
from keras.layers.convolutional import _Pooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Activation, merge, Dense, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.engine import Layer, InputSpec
from keras import activations as activations
from keras import initializations as initializations
from keras import regularizers as regularizers
from keras import constraints as constraints


"""
Helper functions
"""
			
def sparse_pool(input_image, stride = 2, poolsize = (2,2), mode = 'max'):
	pooled_array = []
	counter = 0
	for offset_x in xrange(stride):
		for offset_y in xrange(stride):
			pooled_array +=[max_pool(input_image[:, :, offset_x::stride, offset_y::stride], poolsize, st = (1,1), mode = mode, padding = (0,0), ignore_border = True)]
			counter += 1

	# Concatenate pooled image to create one big image
	running_concat = []
	for it in xrange(stride):
		running_concat += [T.concatenate(pooled_array[stride*it:stride*(it+1)], axis = 3)]
	concatenated_image = T.concatenate(running_concat,axis = 2)

	pooled_output_array = []

	for it in xrange(counter+1):
		pooled_output_array += [T.tensor4()]

	pooled_output_array[0] = concatenated_image

	counter = 0
	for offset_x in xrange(stride):
		for offset_y in xrange(stride):
			pooled_output_array[counter+1] = T.set_subtensor(pooled_output_array[counter][:, :, offset_x::stride, offset_y::stride], pooled_array[counter])
			counter += 1

	return pooled_output_array[counter]

def sparse_W(W_input, stride = 2, filter_shape = (0,0,0,0)):
	W_new = theano.shared(value = np.zeros((filter_shape[0], filter_shape[1], stride*(filter_shape[2]-1)+1, stride*(filter_shape[3]-1)+1),dtype = theano.config.floatX), borrow = True)
	W_new_1 = T.set_subtensor(W_new[:,:,0::stride,0::stride],W_input)
	new_filter_shape = (filter_shape[0], filter_shape[1], stride*(filter_shape[2]-1)+1, stride*(filter_shape[3]-1)+1)
	return W_new_1, new_filter_shape

def conv_output_length(input_length, filter_size, border_mode, stride):
	if input_length is None:
		return None
	assert border_mode in {'same', 'valid'}
	if border_mode == 'same':
		output_length = input_length
	elif border_mode == 'valid':
		output_length = input_length - filter_size + 1
	return (output_length + stride - 1) // stride

def sparse_pool_output_length(input_length, pool_size, stride):
	return input_length - (stride*(pool_size-1)+1)+1

def nikon_getfiles(direc_name,channel_name):
	imglist = os.listdir(direc_name)
	imgfiles = [i for i in imglist if channel_name in i]

	def sorted_nicely(l):
		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		return sorted(l, key = alphanum_key)

	imgfiles = sorted_nicely(imgfiles)
	return imgfiles

def get_image(file_name):
	if '.tif' in file_name:
		im = np.float32(tiff.TIFFfile(file_name).asarray())
	else:
		im = np.float32(imread(file_name))
	return im

def format_coord(x,y,sample_image):
    numrows, numcols = sample_image.shape
    col = int(x+0.5)
    row = int(y+0.5)
    if col>= 0 and col<numcols and row>=0 and row<numrows:
        z = sample_image[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x,y,z)
    else:
        return 'x=%1.4f, y=1.4%f'%(x,y)

"""
Data generator for training_data
"""

def data_generator(channels, batch, pixel_x, pixel_y, labels, win_x = 30, win_y = 30):
	img_list = []
	l_list = []
	for b, x, y, l in zip(batch, pixel_x, pixel_y, labels):
		img = channels[b,:, x-win_x:x+win_x+1, y-win_y:y+win_y+1]
		img_list += [img]
		l_list += [l]

	return np.stack(tuple(img_list),axis = 0), np.array(l_list)

def load_training_data(file_name):
	training_data = np.load(file_name)
	channels = training_data['channels']
	batch = training_data['batch']
	labels = training_data['y']
	pixels_x = training_data['pixels_x']
	pixels_y = training_data['pixels_y']
	win_x = training_data['win_x']
	win_y = training_data['win_y']

	total_batch_size = len(labels)
	num_test = np.int32(np.floor(total_batch_size/10))
	num_train = np.int32(total_batch_size - num_test)
	full_batch_size = np.int32(num_test + num_train)

	"""
	Split data set into training data and validation data
	"""
	arr = np.arange(len(labels))
	arr_shuff = np.random.permutation(arr)

	train_ind = arr_shuff[0:num_train]
	test_ind = arr_shuff[num_train:num_train+num_test]

	X_train, y_train = data_generator(channels, batch[train_ind], pixels_x[train_ind], pixels_y[train_ind], labels[train_ind], win_x = win_x, win_y = win_y)
	X_test, y_test = data_generator(channels, batch[test_ind], pixels_x[test_ind], pixels_y[test_ind], labels[test_ind], win_x = win_x, win_y = win_y)

	return (X_train, y_train), (X_test, y_test)

"""
Keras layers
"""

class sparse_Convolution2D(Layer):
	'''Convolution operator for filtering windows of two-dimensional inputs.
	When using this layer as the first layer in a model,
	provide the keyword argument `input_shape`
	(tuple of integers, does not include the sample axis),
	e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.
	# Examples
	```python
		# apply a 3x3 convolution with 64 output filters on a 256x256 image:
		model = Sequential()
		model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
		# now model.output_shape == (None, 64, 256, 256)
		# add a 3x3 convolution on top, with 32 output filters:
		model.add(Convolution2D(32, 3, 3, border_mode='same'))
		# now model.output_shape == (None, 32, 256, 256)
	```
	# Arguments
		nb_filter: Number of convolution filters to use.
		nb_row: Number of rows in the convolution kernel.
		nb_col: Number of columns in the convolution kernel.
		init: name of initialization function for the weights of the layer
			(see [initializations](../initializations.md)), or alternatively,
			Theano function to use for weights initialization.
			This parameter is only relevant if you don't pass
			a `weights` argument.
		activation: name of activation function to use
			(see [activations](../activations.md)),
			or alternatively, elementwise Theano function.
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: a(x) = x).
		weights: list of numpy arrays to set as initial weights.
		border_mode: 'valid' or 'same'.
		subsample: tuple of length 2. Factor by which to subsample output.
			Also called strides elsewhere.
		W_regularizer: instance of [WeightRegularizer](../regularizers.md)
			(eg. L1 or L2 regularization), applied to the main weights matrix.
		b_regularizer: instance of [WeightRegularizer](../regularizers.md),
			applied to the bias.
		activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
			applied to the network output.
		W_constraint: instance of the [constraints](../constraints.md) module
			(eg. maxnorm, nonneg), applied to the main weights matrix.
		b_constraint: instance of the [constraints](../constraints.md) module,
			applied to the bias.
		dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
			(the depth) is at index 1, in 'tf' mode is it at index 3.
			It defaults to the `image_dim_ordering` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "th".
		bias: whether to include a bias (i.e. make the layer affine rather than linear).
	# Input shape
		4D tensor with shape:
		`(samples, channels, rows, cols)` if dim_ordering='th'
		or 4D tensor with shape:
		`(samples, rows, cols, channels)` if dim_ordering='tf'.
	# Output shape
		4D tensor with shape:
		`(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
		or 4D tensor with shape:
		`(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
		`rows` and `cols` values might have changed due to padding.
	'''
	def __init__(self, nb_filter, nb_row, nb_col,
				 d = 1, init='glorot_uniform', activation='linear', weights=None,
				 border_mode='valid', subsample=(1, 1), dim_ordering=K.image_dim_ordering(),
				 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, **kwargs):

		if border_mode not in {'valid', 'same'}:
			raise Exception('Invalid border mode for Convolution2D:', border_mode)
		self.nb_filter = nb_filter
		self.nb_row = nb_row
		self.nb_col = nb_col
		self.init = initializations.get(init, dim_ordering=dim_ordering)
		self.activation = activations.get(activation)
		self.d = d
		assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
		self.border_mode = border_mode
		self.subsample = tuple(subsample)
		assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
		self.dim_ordering = dim_ordering

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		self.input_spec = [InputSpec(ndim=4)]
		self.initial_weights = weights
		super(sparse_Convolution2D, self).__init__(**kwargs)

	def build(self, input_shape):
		if self.dim_ordering == 'th':
			stack_size = input_shape[1]
			self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
		elif self.dim_ordering == 'tf':
			stack_size = input_shape[3]
			self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
		self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
		self.sparse_W, self.sparse_W_shape = sparse_W(self.W, stride = self.d, filter_shape = self.W_shape)
		self.nb_row_sparse = self.sparse_W_shape[2]
		self.nb_col_sparse = self.sparse_W_shape[3]

		if self.bias:
			self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
			self.trainable_weights = [self.W, self.b]
		else:
			self.trainable_weights = [self.W]
		self.regularizers = []

		if self.W_regularizer:
			self.W_regularizer.set_param(self.W)
			self.regularizers.append(self.W_regularizer)

		if self.bias and self.b_regularizer:
			self.b_regularizer.set_param(self.b)
			self.regularizers.append(self.b_regularizer)

		if self.activity_regularizer:
			self.activity_regularizer.set_layer(self)
			self.regularizers.append(self.activity_regularizer)

		self.constraints = {}
		if self.W_constraint:
			self.constraints[self.W] = self.W_constraint
		if self.bias and self.b_constraint:
			self.constraints[self.b] = self.b_constraint

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def get_output_shape_for(self, input_shape):
		if self.dim_ordering == 'th':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.dim_ordering == 'tf':
			rows = input_shape[1]
			cols = input_shape[2]
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

		rows = conv_output_length(rows, self.nb_row_sparse,
								  self.border_mode, self.subsample[0])
		cols = conv_output_length(cols, self.nb_col_sparse,
								  self.border_mode, self.subsample[1])

		if self.dim_ordering == 'th':
			return (input_shape[0], self.nb_filter, rows, cols)
		elif self.dim_ordering == 'tf':
			return (input_shape[0], rows, cols, self.nb_filter)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

	def call(self, x, mask=None):
		output = K.conv2d(x, self.sparse_W, strides=self.subsample,
						  border_mode=self.border_mode,
						  dim_ordering=self.dim_ordering,
						  filter_shape=self.sparse_W_shape)
		if self.bias:
			if self.dim_ordering == 'th':
				output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
			else:
				raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
		output = self.activation(output)
		return output

	def get_config(self):
		config = {'nb_filter': self.nb_filter,
				  'nb_row': self.nb_row,
				  'nb_col': self.nb_col,
				  'nb_row_sparse': self.nb_row_sparse,
				  'nb_col_sparse': self.nb_col_sparse,
				  'init': self.init.__name__,
				  'activation': self.activation.__name__,
				  'border_mode': self.border_mode,
				  'subsample': self.subsample,
				  'dim_ordering': self.dim_ordering,
				  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
				  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
				  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
				  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
				  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
				  'bias': self.bias}
		base_config = super(sparse_Convolution2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class sparse_MaxPooling2D(Layer):
	'''Max pooling operation for spatial data.
	# Arguments
		pool_size: tuple of 2 integers,
			factors by which to downscale (vertical, horizontal).
			(2, 2) will halve the image in each dimension.
		strides: tuple of 2 integers, or None. Strides values.
		border_mode: 'valid' or 'same'.
			Note: 'same' will only work with TensorFlow for the time being.
		dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
			(the depth) is at index 1, in 'tf' mode is it at index 3.
			It defaults to the `image_dim_ordering` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "th".
	# Input shape
		4D tensor with shape:
		`(samples, channels, rows, cols)` if dim_ordering='th'
		or 4D tensor with shape:
		`(samples, rows, cols, channels)` if dim_ordering='tf'.
	# Output shape
		4D tensor with shape:
		`(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
		or 4D tensor with shape:
		`(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.
	'''

	def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
				 dim_ordering=K.image_dim_ordering(), **kwargs):
		super(sparse_MaxPooling2D, self).__init__(**kwargs)
		self.pool_size = tuple(pool_size)
		if strides is None:
			strides = self.pool_size
		self.strides = tuple(strides)
		assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
		self.border_mode = border_mode
		assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
		self.dim_ordering = dim_ordering
		self.input_spec = [InputSpec(ndim=4)]

	def get_output_shape_for(self, input_shape):
		if self.dim_ordering == 'th':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.dim_ordering == 'tf':
			rows = input_shape[1]
			cols = input_shape[2]
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

		rows = sparse_pool_output_length(rows, self.pool_size[0], self.strides[0])
		cols = sparse_pool_output_length(rows, self.pool_size[1], self.strides[1])

		if self.dim_ordering == 'th':
			return (input_shape[0], input_shape[1], rows, cols)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

	def _pooling_function(self, inputs, pool_size, strides,
						  border_mode, dim_ordering):
		output = sparse_pool(inputs, pool_size = pool_size, stride = strides[0])

	def call(self, x, mask=None):
		output = self._pooling_function(inputs=x, pool_size=self.pool_size,
										strides=self.strides,
										border_mode=self.border_mode,
										dim_ordering=self.dim_ordering)
		return output

	def get_config(self):
		config = {'pool_size': self.pool_size,
				  'border_mode': self.border_mode,
				  'strides': self.strides,
				  'dim_ordering': self.dim_ordering}
		base_config = super(sparse_MaxPooling2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class TensorProd2D(Layer):
	'''Convolution operator for filtering windows of two-dimensional inputs.
	When using this layer as the first layer in a model,
	provide the keyword argument `input_shape`
	(tuple of integers, does not include the sample axis),
	e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.
	# Examples
	```python
		# apply a 3x3 convolution with 64 output filters on a 256x256 image:
		model = Sequential()
		model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
		# now model.output_shape == (None, 64, 256, 256)
		# add a 3x3 convolution on top, with 32 output filters:
		model.add(Convolution2D(32, 3, 3, border_mode='same'))
		# now model.output_shape == (None, 32, 256, 256)
	```
	# Arguments
		nb_filter: Number of convolution filters to use.
		nb_row: Number of rows in the convolution kernel.
		nb_col: Number of columns in the convolution kernel.
		init: name of initialization function for the weights of the layer
			(see [initializations](../initializations.md)), or alternatively,
			Theano function to use for weights initialization.
			This parameter is only relevant if you don't pass
			a `weights` argument.
		activation: name of activation function to use
			(see [activations](../activations.md)),
			or alternatively, elementwise Theano function.
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: a(x) = x).
		weights: list of numpy arrays to set as initial weights.
		border_mode: 'valid' or 'same'.
		subsample: tuple of length 2. Factor by which to subsample output.
			Also called strides elsewhere.
		W_regularizer: instance of [WeightRegularizer](../regularizers.md)
			(eg. L1 or L2 regularization), applied to the main weights matrix.
		b_regularizer: instance of [WeightRegularizer](../regularizers.md),
			applied to the bias.
		activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
			applied to the network output.
		W_constraint: instance of the [constraints](../constraints.md) module
			(eg. maxnorm, nonneg), applied to the main weights matrix.
		b_constraint: instance of the [constraints](../constraints.md) module,
			applied to the bias.
		dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
			(the depth) is at index 1, in 'tf' mode is it at index 3.
			It defaults to the `image_dim_ordering` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "th".
		bias: whether to include a bias (i.e. make the layer affine rather than linear).
	# Input shape
		4D tensor with shape:
		`(samples, channels, rows, cols)` if dim_ordering='th'
		or 4D tensor with shape:
		`(samples, rows, cols, channels)` if dim_ordering='tf'.
	# Output shape
		4D tensor with shape:
		`(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
		or 4D tensor with shape:
		`(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
		`rows` and `cols` values might have changed due to padding.
	'''
	def __init__(self, nb_filter, nb_row, nb_col,
				 init='glorot_uniform', activation='linear', weights=None,
				 border_mode='valid', subsample=(1, 1), dim_ordering=K.image_dim_ordering(),
				 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, **kwargs):

		if border_mode not in {'valid', 'same'}:
			raise Exception('Invalid border mode for Convolution2D:', border_mode)
		self.nb_filter = nb_filter
		self.nb_row = nb_row
		self.nb_col = nb_col
		self.init = initializations.get(init, dim_ordering=dim_ordering)
		self.activation = activations.get(activation)
		assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
		self.border_mode = border_mode
		self.subsample = tuple(subsample)
		assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
		self.dim_ordering = dim_ordering

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		self.input_spec = [InputSpec(ndim=4)]
		self.initial_weights = weights
		super(Convolution2D, self).__init__(**kwargs)

	def build(self, input_shape):
		if self.dim_ordering == 'th':
			stack_size = input_shape[1]
			self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
		elif self.dim_ordering == 'tf':
			stack_size = input_shape[3]
			self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
		self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
		if self.bias:
			self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
			self.trainable_weights = [self.W, self.b]
		else:
			self.trainable_weights = [self.W]
		self.regularizers = []

		if self.W_regularizer:
			self.W_regularizer.set_param(self.W)
			self.regularizers.append(self.W_regularizer)

		if self.bias and self.b_regularizer:
			self.b_regularizer.set_param(self.b)
			self.regularizers.append(self.b_regularizer)

		if self.activity_regularizer:
			self.activity_regularizer.set_layer(self)
			self.regularizers.append(self.activity_regularizer)

		self.constraints = {}
		if self.W_constraint:
			self.constraints[self.W] = self.W_constraint
		if self.bias and self.b_constraint:
			self.constraints[self.b] = self.b_constraint

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def get_output_shape_for(self, input_shape):
		if self.dim_ordering == 'th':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.dim_ordering == 'tf':
			rows = input_shape[1]
			cols = input_shape[2]
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

		rows = conv_output_length(rows, self.nb_row,
								  self.border_mode, self.subsample[0])
		cols = conv_output_length(cols, self.nb_col,
								  self.border_mode, self.subsample[1])

		if self.dim_ordering == 'th':
			return (input_shape[0], self.nb_filter, rows, cols)
		elif self.dim_ordering == 'tf':
			return (input_shape[0], rows, cols, self.nb_filter)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

	def call(self, x, mask=None):

		output = T.tensordot(x, self.W, axes = [1,0]).dimshuffle(0,3,1,2) 

		if self.bias:
			if self.dim_ordering == 'th':
				output += self.b.dimshuffle('x',0,'x','x') 
			else:
				raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
		output = self.activation(output)
		return output

	def get_config(self):
		config = {'nb_filter': self.nb_filter,
				  'nb_row': self.nb_row,
				  'nb_col': self.nb_col,
				  'init': self.init.__name__,
				  'activation': self.activation.__name__,
				  'border_mode': self.border_mode,
				  'subsample': self.subsample,
				  'dim_ordering': self.dim_ordering,
				  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
				  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
				  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
				  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
				  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
				  'bias': self.bias}
		base_config = super(Convolution2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


"""
ResNet helper functions
"""

# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
	def f(input):
		conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
							 init="he_normal", border_mode="same")(input)
		norm = BatchNormalization(mode=0, axis=1)(conv)
		return Activation("relu")(norm)

	return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
	def f(input):
		norm = BatchNormalization(mode=0, axis=1)(input)
		activation = Activation("relu")(norm)
		return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
							 init="he_normal", border_mode="same")(activation)

	return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1)):
	def f(input):
		conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
		conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
		residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
		return _shortcut(input, residual)

	return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
	def f(input):
		conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
		residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
		return _shortcut(input, residual)

	return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
	# Expand channels of shortcut to match residual.
	# Stride appropriately to match residual (width, height)
	# Should be int if network architecture is correctly configured.
	stride_width = input._keras_shape[2] / residual._keras_shape[2]
	stride_height = input._keras_shape[3] / residual._keras_shape[3]
	equal_channels = residual._keras_shape[1] == input._keras_shape[1]

	shortcut = input
	# 1 X 1 conv if shape is different. Else identity.
	if stride_width > 1 or stride_height > 1 or not equal_channels:
		shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
								 subsample=(stride_width, stride_height),
								 init="he_normal", border_mode="valid")(input)

	return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
	def f(input):
		for i in range(repetations):
			init_subsample = (1, 1)
			if i == 0 and not is_first_layer:
				init_subsample = (2, 2)
			input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
		return input

	return f

"""
Siamese Classes
"""

class Siamese(Layer):
	def __init__(self, layer, inputs, merge_mode=None, concat_axis=1, dot_axes=-1):

		if merge_mode not in ['sum', 'mul', 'concat', 'ave', 'join', 'cos', 'dot', None]:
			raise Exception("Invalid merge mode: " + str(mode))

		if merge_mode in {'cos', 'dot'}:
			if len(inputs) > 2:
				raise Exception(mode + " merge takes exactly 2 layers")
			shape1 = inputs[0].output_shape
			shape2 = inputs[1].output_shape
			n1 = len(shape1)
			n2 = len(shape2)
			if mode == 'dot':
				if type(dot_axes) == int:
					if dot_axes < 0:
						dot_axes = [range(dot_axes % n1, n1), range(dot_axes % n2, n2)]
					else:
						dot_axes = [range(n1 - dot_axes, n2), range(1, dot_axes + 1)]
				for i in range(len(dot_axes[0])):
					if shape1[dot_axes[0][i]] != shape2[dot_axes[1][i]]:
						raise Exception(" Dot incompatible layers can not be merged using dot mode")

		self.layer = layer
		self.inputs = inputs
		self.params = []
		self.merge_mode = merge_mode
		self.concat_axis = concat_axis
		self.dot_axes = dot_axes
		layer.set_previous(inputs[0])
		self.regularizers = []
		self.constraints = []
		self.updates = []
		layers = [layer] + inputs
		for l in layers:
			params, regs, consts, updates = l.get_params()
			self.regularizers += regs
			self.updates += updates
			# params and constraints have the same size
			for p, c in zip(params, consts):
				if p not in self.params:
					self.params.append(p)
					self.constraints.append(c)

	@property
	def output_shape(self):
		if merge_mode is None:
			return self.layer.output_shape
		input_shapes = [self.layer.output_shape]*len(self.inputs)
		if self.merge_mode in ['sum', 'mul', 'ave']:
			return input_shapes[0]
		elif self.merge_mode == 'concat':
			output_shape = list(input_shapes[0])
			for shape in input_shapes[1:]:
				output_shape[self.concat_axis] += shape[self.concat_axis]
			return tuple(output_shape)
		elif self.merge_mode == 'join':
			return None
		elif self.merge_mode == 'dot':
			shape1 = list(input_shapes[0])
			shape2 = list(input_shapes[1])
			for i in self.dot_axes[0]:
				shape1.pop(i)
			for i in self.dot_axes[1]:
				shape2.pop(i)
			shape = shape1 + shape2[1:]
			if len(shape) == 1:
				shape.append(1)
			return tuple(shape)
		elif self.merge_mode == 'cos':
			return tuple(input_shapes[0][0], 1)

	def get_params(self):
		return self.params, self.regularizers, self.constraints, self.updates

	def get_output_at(self, head, train=False):
		self.layer.set_previous(self.inputs[head])
		return self.layer.get_output(train)

	def get_output_join(self, train=False):
		o = OrderedDict()
		for i in range(len(inputs)):
			X = self.get_output_at(i, train)
			if X.name is None:
				raise ValueError("merge_mode='join' only works with named inputs")
			o[X.name] = X
		return o

	def get_output_sum(self, train=False):
		s = self.get_output_at(0, train)
		for i in range(1, len(self.inputs)):
			s += self.get_output_at(i, train)
		return s

	def get_output_ave(self, train=False):
		n = len(self.inputs)
		s = self.get_output_at(0, train)
		for i in range(1, n):
			s += self.get_output_at(i, train)
		s /= n
		return s

	def get_output_concat(self, train=False):
		inputs = [self.get_output_at(i, train) for i in range(len(self.inputs))]
		return T.concatenate(inputs, axis=self.concat_axis)

	def get_output_mul(self, train=False):
		s = self.get_output_at(0, train)
		for i in range(1, len(self.inputs)):
			s *= self.get_output_at(i, train)
		return s

	def get_output_dot(self, train=False):
		l1 = self.get_output_at(0, train)
		l2 = self.get_output_at(1, train)
		output = T.batched_tensordot(l1, l2, self.dot_axes)
		output = output.dimshuffle((0, 'x'))
		return output

	def get_output_cos(self, train=False):
		l1 = self.get_output_at(0, train)
		l2 = self.get_output_at(1, train)
		output, _ = theano.scan(lambda v1, v2: T.dot(v1, v2)/T.sqrt(T.dot(v1, v1) * T.dot(v2, v2)), sequences=[l1, l2], outputs_info=None)
		output = output.dimshuffle((0, 'x'))
		return output

	def get_output(self, train=False):
		mode = self.merge_mode
		if mode == 'join':
			return self.get_output_join(train)
		elif mode == 'concat':
			return self.get_output_concat(train)
		elif mode == 'sum':
			return self.get_output_sum(train)
		elif mode == 'ave':
			return self.get_output_ave(train)
		elif mode == 'mul':
			return self.get_output_mul(train)
		elif mode == 'dot':
			return self.get_output_dot(train)
		elif mode == 'cos':
			return self.get_output_dot(train)

	def get_input(self, train=False):
		res = []
		for i in range(len(self.inputs)):
			o = self.inputs[i].get_input(train)
			if not type(o) == list:
				o = [o]
			for output in o:
				if output not in res:
					res.append(output)
		return res

	@property
	def input(self):
		return self.get_input()

	def supports_masked_input(self):
		return False

	def get_output_mask(self, train=None):
		return None

	def get_weights(self):
		weights = layer.get_weights()
		for m in self.inputs:
			weights += m.get_weights()
		return weights

	def set_weights(self, weights):
		nb_param = len(self.layer.params)
		self.layer.set_weights(weights[:nb_param])
		weights = weights[nb_param:]
		for i in range(len(self.inputs)):
			nb_param = len(self.inputs[i].params)
			self.inputs[i].set_weights(weights[:nb_param])
			weights = weights[nb_param:]

	def get_config(self):

		config = {"name": self.__class__.__name__,
				  "layer": self.layer.get_config,
				  "inputs": [m.get_config() for m in self.inputs],
				  "merge_mode": self.merge_mode,
				  "concat_axis": self.concat_axis,
				  "dot_axes": self.dot_axes
				  }
		base_config = super(Siamese, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
		
class SiameseHead(Layer):
	def __init__(self, head):
		self.head = head
		self.params = []
	def get_output(self, train=False):
		return self.get_input(train)

	@property
	def input_shape(self):
		return self.previous.layer.output_shape

	def get_input(self, train=False):
		return self.previous.get_output_at(self.head, train)

	def get_config(self):

		config = {"name": self.__class__.__name__,
				  "head": self.head
				  }

		base_config = super(SiameseHead, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def set_previous(self, layer):
		self.previous = layer

def add_shared_layer(layer,inputs):
	input_layers = [l.layers[-1] for l in inputs]
	s = Siamese(layer, input_layers)
	for i in range(len(inputs)):
		sh = SiameseHead(i)
		inputs[i].add (s)
		inputs[i].add(sh)