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
from scipy import ndimage
import threading
import scipy.ndimage as ndi
from scipy import linalg
import re
import random
import itertools
import h5py
import datetime

from theano.tensor.nnet import conv
from theano.tensor.signal.pool import pool_2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, flip_axis, array_to_img, img_to_array, load_img, list_pictures
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.engine import Layer, InputSpec
from keras.utils import np_utils
from keras import activations as activations
from keras import initializations as initializations
from keras import regularizers as regularizers
from keras import constraints as constraints


"""
Helper functions
"""

def set_weights(model, weights_path):
	f = h5py.File(weights_path ,'r')

	# for key in f.keys():
	# 	g = f[key]
	# 	weights = [g[k] for k in g.keys()]
	# 	print weights

	for layer in model.layers:
		if 'tensorprod2d' in layer.name:
			idsplit = layer.name.split('_')[-1]
			layer.name = 'dense_' + idsplit

		if 'sparse_convolution2d' in layer.name:
			idsplit = layer.name.split('_')[-1]
			layer.name = 'convolution2d_' + idsplit

		if layer.name in f.keys():
			if 'bn' in layer.name:
				g = f[layer.name]
				keys = ['{}_gamma'.format(layer.name), '{}_beta'.format(layer.name), '{}_running_mean'.format(layer.name), '{}_running_std'.format(layer.name)]
				weights = [g[key] for key in keys]
				layer.set_weights(weights)

			if 'batch' in layer.name:
				g = f[layer.name]
				keys = ['{}_gamma'.format(layer.name), '{}_beta'.format(layer.name), '{}_running_mean'.format(layer.name), '{}_running_std'.format(layer.name)]
				weights = [g[key] for key in keys]
				layer.set_weights(weights)

			else:
				g = f[layer.name]
				weights = [g[key] for key in g.keys()]
				layer.set_weights(weights)

	return model

def rotate_array_0(arr):
	return arr

def rotate_array_90(arr):
	axes_order = range(arr.ndim - 2) + [arr.ndim-1, arr.ndim-2]
	slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None),slice(None,None,-1)]
	return arr[tuple(slices)].transpose(axes_order)

def rotate_array_180(arr):
	slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None,None,-1), slice(None,None,-1)]
	return arr[tuple(slices)]

def rotate_array_270(arr):
	axes_order = range(arr.ndim-2) + [arr.ndim-1, arr.ndim-2]
	slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None,None,-1), slice(None)]
	return arr[tuple(slices)].transpose(axes_order)

def categorical_sum(y_true, y_pred):
    return K.sum(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred*0, axis=-1)))

def rate_scheduler(lr = .001, decay = 0.95):
	def output_fn(epoch):
		epoch = np.int(epoch)
		new_lr = lr * (decay ** epoch)
		return new_lr
	return output_fn

def euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
	'''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	margin = 1
	return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def morgan_loss(y_true, y_pred, Q = 1e5):
	return K.mean(y_true*2/Q*K.square(y_pred) + (1-y_true)*2*Q*K.exp(-2.77/Q*K.abs(y_pred)))

def same_loss(y_true, y_pred):
	return y_pred

def combinations_diff(array):
	array = list(array)
	it = itertools.combinations(array,2)

	combs = []
	y = []

	for subset in it:
		combs += [subset]
		y += [0]

	return combs, y

def combinations_same(array):
	combs = []
	y = []
	for elt in array:
		for j in xrange((len(array)-1)/2):
			combs += [(elt,elt)]
			y += [1]
	return combs, y

def combinations(array):
	if len(array) % 2 == 0:
		array = array[0:-1]
	comb_diff, y_diff = combinations_diff(array)
	comb_same, y_same = combinations_same(array)

	comb = comb_diff + comb_same
	y = y_diff + y_same

	dual = list(zip(comb,y))
	random.shuffle(dual)

	comb, y = zip(*dual)

	return comb, np.array(list(y))

def process_image(channel_img, win_x, win_y, std = False):
	if std:
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
		std = np.std(channel_img)
		channel_img /= std
		return channel_img
	else:
		p50 = np.percentile(channel_img, 50)
		channel_img /= p50
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
		return channel_img

def tensorprod_softmax(x):
	e_output = T.exp(x - x.max(axis = 1, keepdims=True))
	softmax = e_output/e_output.sum(axis = 1, keepdims = True)
	return softmax

def sparse_pool(input_image, stride = 2, pool_size = (2,2), mode = 'max'):
	pooled_array = []
	counter = 0
	for offset_x in xrange(stride):
		for offset_y in xrange(stride):
			pooled_array +=[pool_2d(input_image[:, :, offset_x::stride, offset_y::stride], pool_size, st = (1,1), mode = mode, padding = (0,0), ignore_border = True)]
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
	return input_length - stride*(pool_size-1)

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

def randomly_rotate_array(x):
	r = np.random.random()
	if r < 0.25:
		x = rotate_array_0(x)
	if r > 0.25 and r < 0.5:
		x = rotate_array_90(x)
	if r > 0.5 and r < 0.75:
		x = rotate_array_180(x)
	if r > 0.75:
		x = rotate_array_270(x)

	return x

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

def get_data_sample(file_name):
	training_data = np.load(file_name)
	channels = training_data["channels"]
	batch = training_data["batch"]
	labels = training_data["y"]
	pixels_x = training_data["pixels_x"]
	pixels_y = training_data["pixels_y"]
	win_x = training_data["win_x"]
	win_y = training_data["win_y"]

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

	X_test, y_test = data_generator(channels.astype("float32"), batch[test_ind], pixels_x[test_ind], pixels_y[test_ind], labels[test_ind], win_x = win_x, win_y = win_y)
	train_dict = {"channels": channels.astype("float32"), "batch": batch[train_ind], "pixels_x": pixels_x[train_ind], "pixels_y": pixels_y[train_ind], "labels": labels[train_ind], "win_x": win_x, "win_y": win_y}
	
	return train_dict, (X_test, y_test)

def get_data_siamese(file_name):
	training_data = np.load(file_name)
	image_list = training_data["image_list"]
	id_list = training_data["id_list"]

	id_combs, labels = combinations(id_list)

	total_batch_size = len(labels)
	num_test = np.int32(np.floor(total_batch_size/200))
	num_train = np.int32(total_batch_size - num_test)
	full_batch_size = np.int32(num_test + num_train)

	id_train = id_combs[0:num_train]
	label_train = labels[0:num_train]

	id_test = id_combs[num_train:num_train+num_test]
	label_test = labels[num_train:num_train+num_test]
	input_1_test = np.zeros((num_test,) + image_list.shape[1:], dtype = 'float32')
	input_2_test = np.zeros((num_test,) + image_list.shape[1:], dtype = 'float32')

	for j in xrange(num_test):
		input_1_test[j] = image_list[id_test[j][0]]
		input_2_test[j] = image_list[id_test[j][1]]

	train_dict = {"image_list": image_list, "ids": id_train, "labels": label_train}
	test_input_dict = {"input_1":input_1_test, "input_2":input_2_test}
	test_label_dict = {"lambda_1": label_test}
	return train_dict, (test_input_dict, test_label_dict)

class Iterator(object):

	def __init__(self, N, batch_size, shuffle, seed):
		self.N = N
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.batch_index = 0
		self.total_batches_seen = 0
		self.lock = threading.Lock()
		self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

	def reset(self):
		self.batch_index = 0

	def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
		# ensure self.batch_index is 0
		self.reset()
		while 1:
			if self.batch_index == 0:
				index_array = np.arange(N)
				if shuffle:
					if seed is not None:
						np.random.seed(seed + self.total_batches_seen)
					index_array = np.random.permutation(N)

			current_index = (self.batch_index * batch_size) % N
			if N >= current_index + batch_size:
				current_batch_size = batch_size
				self.batch_index += 1
			else:
				current_batch_size = N - current_index
				self.batch_index = 0
			self.total_batches_seen += 1
			yield (index_array[current_index: current_index + current_batch_size],
				   current_index, current_batch_size)

	def __iter__(self):
		# needed if we want to do something like:
		# for x, y in data_gen.flow(...):
		return self

	def __next__(self, *args, **kwargs):
		return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):

	def __init__(self, X, y, image_data_generator,
				 batch_size=32, shuffle=False, seed=None,
				 dim_ordering=K.image_dim_ordering(),
				 save_to_dir=None, save_prefix='', save_format='jpeg'):
		if y is not None and len(X) != len(y):
			raise Exception('X (images tensor) and y (labels) '
							'should have the same length. '
							'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
		self.X = X
		self.y = y
		self.image_data_generator = image_data_generator
		self.dim_ordering = dim_ordering
		self.save_to_dir = save_to_dir
		self.save_prefix = save_prefix
		self.save_format = save_format
		super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

	def next(self):
		# for python 2.x.
		# Keeps under lock only the mechanism which advances
		# the indexing of each batch
		# see http://anandology.com/blog/using-iterators-and-generators/
		with self.lock:
			index_array, current_index, current_batch_size = next(self.index_generator)
		# The transformation of images is not under thread lock so it can be done in parallel
		batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
		for i, j in enumerate(index_array):
			x = self.X[j]
			x = self.image_data_generator.random_transform(x.astype('float32'))
			x = self.image_data_generator.standardize(x)
			batch_x[i] = x
		if self.save_to_dir:
			for i in range(current_batch_size):
				img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
				fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
																  index=current_index + i,
																  hash=np.random.randint(1e4),
																  format=self.save_format)
				img.save(os.path.join(self.save_to_dir, fname))
		if self.y is None:
			return batch_x
		batch_y = self.y[index_array]
		return batch_x, batch_y

class SiameseNumpyArrayIterator(Iterator):

	def __init__(self, train_dict, image_data_generator,
				 batch_size=32, shuffle=False, seed=None,
				 dim_ordering=K.image_dim_ordering(),
				 save_to_dir=None, save_prefix='', save_format='jpeg'):
		if train_dict["labels"] is not None and len(train_dict["ids"]) != len(train_dict["labels"]):
			raise Exception('ids (location of pairs of images) and y (labels) '
							'should have the same length. '
							'Found: ids.shape = %s, y.shape = %s' % (np.asarray(train_dict["ids"]).shape, np.asarray(train_dict["labels"]).shape))
		self.X = train_dict["image_list"]
		self.ids = train_dict["ids"]
		self.y = train_dict["labels"]
		self.image_data_generator = image_data_generator
		self.dim_ordering = dim_ordering
		self.save_to_dir = save_to_dir
		self.save_prefix = save_prefix
		self.save_format = save_format
		super(SiameseNumpyArrayIterator, self).__init__(len(train_dict["ids"]), batch_size, shuffle, seed)

	def next(self):
		# for python 2.x.
		# Keeps under lock only the mechanism which advances
		# the indexing of each batch
		# see http://anandology.com/blog/using-iterators-and-generators/
		with self.lock:
			index_array, current_index, current_batch_size = next(self.index_generator)
		# The transformation of images is not under thread lock so it can be done in parallel
		batch_x1 = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
		batch_x2 = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
		for i, j in enumerate(index_array):
			ind = self.ids[j]
			x1 = self.X[ind[0]]
			x1 = self.image_data_generator.random_transform(x1.astype('float32'))
			x1 = self.image_data_generator.standardize(x1)

			x2 = self.X[ind[1]]
			x2 = self.image_data_generator.random_transform(x2.astype('float32'))
			x2 = self.image_data_generator.standardize(x2)

			batch_x1[i] = x1
			batch_x2[i] = x2
		if self.save_to_dir:
			for i in range(current_batch_size):
				img = array_to_img(batch_x1[i], self.dim_ordering, scale=True)
				fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix +'1',
																  index=current_index + i,
																  hash=np.random.randint(1e4),
																  format=self.save_format)
				img.save(os.path.join(self.save_to_dir, fname))

				img = array_to_img(batch_x2[i], self.dim_ordering, scale=True)
				fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix +'2',
																  index=current_index + i,
																  hash=np.random.randint(1e4),
																  format=self.save_format)
				img.save(os.path.join(self.save_to_dir, fname))
		if self.y is None:
			return {'input_1': batch_x1, 'input_2': batch_x2}
		batch_y = self.y[index_array]
		return {'input_1': batch_x1, 'input_2': batch_x2}, {'lambda_1': batch_y}


class ImageSampleArrayIterator(Iterator):

	def __init__(self, train_dict, image_data_generator,
				 batch_size=32, shuffle=False, seed=None,
				 dim_ordering=K.image_dim_ordering(),
				 save_to_dir=None, save_prefix='', save_format='jpeg'):

		if train_dict["labels"] is not None and len(train_dict["pixels_x"]) != len(train_dict["labels"]):
			raise Exception('Number of sampled pixels and y (labels) '
							'should have the same length. '
							'Found: Number of sampled pixels = %s, y.shape = %s' % (len(train_dict["pixels_x"]), np.asarray(train_dict["labels"]).shape))

		self.X = train_dict["channels"]
		self.y = train_dict["labels"]
		self.b = train_dict["batch"]
		self.pixels_x = train_dict["pixels_x"]
		self.pixels_y = train_dict["pixels_y"]
		self.win_x = train_dict["win_x"]
		self.win_y = train_dict["win_y"]
		self.image_data_generator = image_data_generator
		self.dim_ordering = dim_ordering
		self.save_to_dir = save_to_dir
		self.save_prefix = save_prefix
		self.save_format = save_format
		super(ImageSampleArrayIterator, self).__init__(len(train_dict["labels"]), batch_size, shuffle, seed)

	def next(self):
		# for python 2.x.
		# Keeps under lock only the mechanism which advances
		# the indexing of each batch
		# see http://anandology.com/blog/using-iterators-and-generators/
		with self.lock:
			index_array, current_index, current_batch_size = next(self.index_generator)
		# The transformation of images is not under thread lock so it can be done in parallel
		batch_x = np.zeros(tuple([current_batch_size] + [self.X.shape[1]] + [2*self.win_x +1, 2*self.win_y+1]))
		for i, j in enumerate(index_array):
			x = self.X[self.b[j],:,self.pixels_x[j]-self.win_x:self.pixels_x[j]+self.win_x+1, self.pixels_y[j]-self.win_y:self.pixels_y[j]+self.win_y+1]
			x = self.image_data_generator.random_transform(x.astype('float32'))
			x = self.image_data_generator.standardize(x)
			batch_x[i] = x
		if self.save_to_dir:
			for i in range(current_batch_size):
				img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
				fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
																  index=current_index + i,
																  hash=np.random.randint(1e4),
																  format=self.save_format)
				img.save(os.path.join(self.save_to_dir, fname))
		if self.y is None:
			return batch_x
		batch_y = self.y[index_array]
		return batch_x, batch_y

class ImageDataGenerator(object):
	'''Generate minibatches with
	real-time data augmentation.
	# Arguments
		featurewise_center: set input mean to 0 over the dataset.
		samplewise_center: set each sample mean to 0.
		featurewise_std_normalization: divide inputs by std of the dataset.
		samplewise_std_normalization: divide each input by its std.
		zca_whitening: apply ZCA whitening.
		rotation_range: degrees (0 to 180).
		width_shift_range: fraction of total width.
		height_shift_range: fraction of total height.
		shear_range: shear intensity (shear angle in radians).
		zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
			in the range [1-z, 1+z]. A sequence of two can be passed instead
			to select this range.
		channel_shift_range: shift range for each channels.
		fill_mode: points outside the boundaries are filled according to the
			given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
			is 'nearest'.
		cval: value used for points outside the boundaries when fill_mode is
			'constant'. Default is 0.
		horizontal_flip: whether to randomly flip images horizontally.
		vertical_flip: whether to randomly flip images vertically.
		rescale: rescaling factor. If None or 0, no rescaling is applied,
			otherwise we multiply the data by the value provided (before applying
			any other transformation).
		dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
			(the depth) is at index 1, in 'tf' mode it is at index 3.
			It defaults to the `image_dim_ordering` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "th".
	'''
	def __init__(self,
				 featurewise_center=False,
				 samplewise_center=False,
				 featurewise_std_normalization=False,
				 samplewise_std_normalization=False,
				 zca_whitening=False,
				 rotation_range=0.,
				 rotate = False,
				 width_shift_range=0.,
				 height_shift_range=0.,
				 shear_range=0.,
				 zoom_range=0.,
				 channel_shift_range=0.,
				 fill_mode='nearest',
				 cval=0.,
				 horizontal_flip=False,
				 vertical_flip=False,
				 rescale=None,
				 dim_ordering=K.image_dim_ordering()):
		self.__dict__.update(locals())
		self.mean = None
		self.std = None
		self.principal_components = None
		self.rescale = rescale

		if dim_ordering not in {'tf', 'th'}:
			raise Exception('dim_ordering should be "tf" (channel after row and '
							'column) or "th" (channel before row and column). '
							'Received arg: ', dim_ordering)
		self.dim_ordering = dim_ordering
		if dim_ordering == 'th':
			self.channel_index = 1
			self.row_index = 2
			self.col_index = 3
		if dim_ordering == 'tf':
			self.channel_index = 3
			self.row_index = 1
			self.col_index = 2

		if np.isscalar(zoom_range):
			self.zoom_range = [1 - zoom_range, 1 + zoom_range]
		elif len(zoom_range) == 2:
			self.zoom_range = [zoom_range[0], zoom_range[1]]
		else:
			raise Exception('zoom_range should be a float or '
							'a tuple or list of two floats. '
							'Received arg: ', zoom_range)

	def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
			 save_to_dir=None, save_prefix='', save_format='jpeg'):
		return NumpyArrayIterator(
			X, y, self,
			batch_size=batch_size, shuffle=shuffle, seed=seed,
			dim_ordering=self.dim_ordering,
			save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

	def sample_flow(self, train_dict, batch_size=32, shuffle=True, seed=None,
			 save_to_dir=None, save_prefix='', save_format='jpeg'):
		return ImageSampleArrayIterator(
			train_dict, self,
			batch_size=batch_size, shuffle=shuffle, seed=seed,
			dim_ordering=self.dim_ordering,
			save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

	def siamese_flow(self, train_dict, batch_size=32, shuffle=True, seed=None,
			 save_to_dir=None, save_prefix='', save_format='jpeg'):
		return SiameseNumpyArrayIterator(
			train_dict, self,
			batch_size=batch_size, shuffle=shuffle, seed=seed,
			dim_ordering=self.dim_ordering,
			save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

	def flow_from_directory(self, directory,
							target_size=(256, 256), color_mode='rgb',
							classes=None, class_mode='categorical',
							batch_size=32, shuffle=True, seed=None,
							save_to_dir=None, save_prefix='', save_format='jpeg'):
		return DirectoryIterator(
			directory, self,
			target_size=target_size, color_mode=color_mode,
			classes=classes, class_mode=class_mode,
			dim_ordering=self.dim_ordering,
			batch_size=batch_size, shuffle=shuffle, seed=seed,
			save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

	def standardize(self, x):
		if self.rescale:
			x *= self.rescale
		# x is a single image, so it doesn't have image number at index 0
		img_channel_index = self.channel_index - 1
		if self.samplewise_center:
			x -= np.mean(x, axis=img_channel_index, keepdims=True)
		if self.samplewise_std_normalization:
			x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

		if self.featurewise_center:
			x -= self.mean
		if self.featurewise_std_normalization:
			x /= (self.std + 1e-7)

		if self.zca_whitening:
			flatx = np.reshape(x, (x.size))
			whitex = np.dot(flatx, self.principal_components)
			x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

		return x

	def random_transform(self, x):
		# x is a single image, so it doesn't have image number at index 0
		img_row_index = self.row_index - 1
		img_col_index = self.col_index - 1
		img_channel_index = self.channel_index - 1

		# use composition of homographies to generate final transform that needs to be applied
		if self.rotation_range:
			theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
		else:
			theta = 0
		rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
									[np.sin(theta), np.cos(theta), 0],
									[0, 0, 1]])
		if self.height_shift_range:
			tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
		else:
			tx = 0

		if self.width_shift_range:
			ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
		else:
			ty = 0

		translation_matrix = np.array([[1, 0, tx],
									   [0, 1, ty],
									   [0, 0, 1]])
		if self.shear_range:
			shear = np.random.uniform(-self.shear_range, self.shear_range)
		else:
			shear = 0
		shear_matrix = np.array([[1, -np.sin(shear), 0],
								 [0, np.cos(shear), 0],
								 [0, 0, 1]])

		if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
			zx, zy = 1, 1
		else:
			zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
		zoom_matrix = np.array([[zx, 0, 0],
								[0, zy, 0],
								[0, 0, 1]])

		transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

		h, w = x.shape[img_row_index], x.shape[img_col_index]
		transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
		x = apply_transform(x, transform_matrix, img_channel_index,
							fill_mode=self.fill_mode, cval=self.cval)
		if self.channel_shift_range != 0:
			x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

		if self.horizontal_flip:
			if np.random.random() < 0.5:
				x = flip_axis(x, img_col_index)

		if self.vertical_flip:
			if np.random.random() < 0.5:
				x = flip_axis(x, img_row_index)

		if self.rotate:
			r = np.random.random()
			if r < 0.25:
				x = rotate_array_0(x)
			if r > 0.25 and r < 0.5:
				x = rotate_array_90(x)
			if r > 0.5 and r < 0.75:
				x = rotate_array_180(x)
			if r > 0.75:
				x = rotate_array_270(x)
		# TODO:
		# channel-wise normalization
		# barrel/fisheye
		return x

	def fit(self, X,
			augment=False,
			rounds=1,
			seed=None):
		'''Required for featurewise_center, featurewise_std_normalization
		and zca_whitening.
		# Arguments
			X: Numpy array, the data to fit on.
			augment: whether to fit on randomly augmented samples
			rounds: if `augment`,
				how many augmentation passes to do over the data
			seed: random seed.
		'''
		X = np.copy(X)
		if augment:
			aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
			for r in range(rounds):
				for i in range(X.shape[0]):
					aX[i + r * X.shape[0]] = self.random_transform(X[i])
			X = aX

		if self.featurewise_center:
			self.mean = np.mean(X, axis=0)
			X -= self.mean

		if self.featurewise_std_normalization:
			self.std = np.std(X, axis=0)
			X /= (self.std + 1e-7)

		if self.zca_whitening:
			flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
			sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
			U, S, V = linalg.svd(sigma)
			self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
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
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

		rows = sparse_pool_output_length(rows, self.pool_size[0], self.strides[0])
		cols = sparse_pool_output_length(cols, self.pool_size[1], self.strides[1])

		if self.dim_ordering == 'th':
			return (input_shape[0], input_shape[1], rows, cols)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

	def _pooling_function(self, inputs, pool_size, strides,
						  border_mode, dim_ordering):
		output = sparse_pool(inputs, pool_size = pool_size, stride = strides[0])
		return output

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
	def __init__(self, input_dim, output_dim,
				 init='glorot_uniform', activation='linear', weights=None,
				 border_mode='valid', subsample=(1, 1), dim_ordering=K.image_dim_ordering(),
				 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, **kwargs):

		if border_mode not in {'valid', 'same'}:
			raise Exception('Invalid border mode for Convolution2D:', border_mode)
		self.input_dim = input_dim
		self.output_dim = output_dim
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
		super(TensorProd2D, self).__init__(**kwargs)

	def build(self, input_shape):
		if self.dim_ordering == 'th':
			self.W_shape = (self.input_dim, self.output_dim)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
		self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
		if self.bias:
			self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
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
			return(input_shape[0], self.output_dim, rows, cols)
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
		config = {'input_dim': self.input_dim,
				  'output_dim': self.output_dim,
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
		base_config = super(TensorProd2D, self).get_config()
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
Training convnets
"""

def train_model_sample(model = None, dataset = None,  optimizer = None, 
	expt = "", it = 0, batch_size = 32, n_epoch = 100,
	direc_save = "/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/", 
	direc_data = "/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/", 
	lr_sched = rate_scheduler(lr = 0.01, decay = 0.95),
	rotate = True, flip = True, shear = 0):

	training_data_file_name = os.path.join(direc_data, dataset + ".npz")
	todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

	file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

	train_dict, (X_test, Y_test) = get_data_sample(training_data_file_name)

	# the data, shuffled and split between train and test sets
	print('X_train shape:', train_dict["channels"].shape)
	print(train_dict["pixels_x"].shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	# determine the number of classes
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[-1]

	# convert class vectors to binary class matrices
	train_dict["labels"] = np_utils.to_categorical(train_dict["labels"], n_classes)
	Y_test = np_utils.to_categorical(Y_test, n_classes)

	model.compile(loss='categorical_crossentropy',
				  optimizer=optimizer,
				  metrics=['accuracy'])

	print('Using real-time data augmentation.')

	# this will do preprocessing and realtime data augmentation
	datagen = ImageDataGenerator(
		rotate = rotate,  # randomly rotate images by 90 degrees
		shear_range = shear, # randomly shear images in the range (radians , -shear_range to shear_range)
		horizontal_flip= flip,  # randomly flip images
		vertical_flip= flip)  # randomly flip images

	# fit the model on the batches generated by datagen.flow()
	loss_history = model.fit_generator(datagen.sample_flow(train_dict, batch_size=batch_size),
						samples_per_epoch=len(train_dict["labels"]),
						nb_epoch=n_epoch,
						validation_data=(X_test, Y_test),
						callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto'),
							LearningRateScheduler(lr_sched)])

	np.savez(file_name_save_loss, loss_history = loss_history.history)

"""
Executing convnets
"""

def get_image_sizes(data_location, channel_names):
	img_list_channels = []
	for channel in channel_names:
		img_list_channels += [nikon_getfiles(data_location, channel)]
	img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

	return img_temp.shape
	
def get_images_from_directory(data_location, channel_names):
	img_list_channels = []
	for channel in channel_names:
		img_list_channels += [nikon_getfiles(data_location, channel)]

	img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

	n_channels = len(channel_names)
	all_images = []

	for stack_iteration in xrange(len(img_list_channels[0])):
		all_channels = np.zeros((1, n_channels, img_temp.shape[0],img_temp.shape[1]), dtype = 'float32')
		for j in xrange(n_channels):
			channel_img = get_image(os.path.join(data_location, img_list_channels[j][stack_iteration]))
			all_channels[0,j,:,:] = channel_img
		all_images += [all_channels]
	
	return all_images

def run_model(image, model, win_x = 30, win_y = 30, std = False, split = True, process = True):

	if process:
		for j in xrange(image.shape[1]):
			image[0,j,:,:] = process_image(image[0,j,:,:], win_x, win_y, std)

	if split:
		image_size_x = image.shape[2]/2
		image_size_y = image.shape[3]/2
	else:
		image_size_x = image.shape[2]
		image_size_y = image.shape[3]

	evaluate_model = K.function(
		[model.layers[0].input, K.learning_phase()],
		[model.layers[-1].output]
		) 

	n_features = model.layers[-1].output_shape[1]

	if split:
		model_output = np.zeros((n_features,2*image_size_x-win_x*2, 2*image_size_y-win_y*2), dtype = 'float32')

		img_0 = image[:,:, 0:image_size_x+win_x, 0:image_size_y+win_y]
		img_1 = image[:,:, 0:image_size_x+win_x, image_size_y-win_y:]
		img_2 = image[:,:, image_size_x-win_x:, 0:image_size_y+win_y]
		img_3 = image[:,:, image_size_x-win_x:, image_size_y-win_y:]

		model_output[:, 0:image_size_x-win_x, 0:image_size_y-win_y] = evaluate_model([img_0, 0])[0]
		model_output[:, 0:image_size_x-win_x, image_size_y-win_y:] = evaluate_model([img_1, 0])[0]
		model_output[:, image_size_x-win_x:, 0:image_size_y-win_y] = evaluate_model([img_2, 0])[0]
		model_output[:, image_size_x-win_x:, image_size_y-win_y:] = evaluate_model([img_3, 0])[0]

	else:
		model_output = evaluate_model([image,0])[0]

	model_output = np.pad(model_output, pad_width = [(0,0), (win_x, win_x),(win_y,win_y)], mode = 'constant', constant_values = [(0,0), (0,0), (0,0)])
	return model_output

def run_model_on_directory(data_location, channel_names, output_location, model, win_x = 30, win_y = 30, std = False, split = True, process = True, save = True):
	n_features = model.layers[-1].output_shape[1]
	counter = 0

	image_list = get_images_from_directory(data_location, channel_names)
	processed_image_list = []

	for image in image_list:
		print "Processing image " + str(counter + 1) + " of " + str(len(image_list))
		processed_image = run_model(image, model, win_x = win_x, win_y = win_y, std = std, split = split, process = process)
		processed_image_list += [processed_image]

		# Save images
		if save:
			for feat in xrange(n_features):
				feature = processed_image[feat,:,:]
				cnnout_name = os.path.join(output_location, 'feature_' + str(feat) +"_frame_"+ str(counter) + r'.tif')
				tiff.imsave(cnnout_name,feature)
		counter += 1

	return processed_image_list


