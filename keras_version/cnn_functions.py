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

from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology as morph
from pywt import WaveletPacket2D
from skimage.transform import resize
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread
from skimage.filters import threshold_otsu
import skimage as sk
from sklearn.utils.linear_assignment_ import linear_assignment

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
try:
	from keras import initializations as initializations
except ImportError:
	from keras import initializers as initializations
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
	
	# print f['model_weights'].keys()	

	for layer in model.layers:
		if 'tensorprod2d' in layer.name:
			idsplit = layer.name.split('_')[-1]
			layer.name = 'dense_' + idsplit

		if 'sparse_convolution2d' in layer.name:
			idsplit = layer.name.split('_')[-1]
			layer.name = 'convolution2d_' + idsplit

	for layer in model.layers:
		if 'model_weights' in f.keys():
			if layer.name in f['model_weights'].keys():
				if 'bn' in layer.name:
					g = f['model_weights'][layer.name]
					keys = ['{}_gamma'.format(layer.name), '{}_beta'.format(layer.name), '{}_running_mean'.format(layer.name), '{}_running_std'.format(layer.name)]
					weights = [g[key] for key in keys]
					layer.set_weights(weights)

				if 'batch' in layer.name:
					g = f['model_weights'][layer.name]
					keys = ['{}_gamma'.format(layer.name), '{}_beta'.format(layer.name), '{}_running_mean'.format(layer.name), '{}_running_std'.format(layer.name)]
					weights = [g[key] for key in keys]
					layer.set_weights(weights)

				else:
					g = f['model_weights'][layer.name]
					weights = [g[key] for key in g.keys()]
					layer.set_weights(weights)
		else:
			# In case old keras saving convention is used
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

def form_coord(x,y,sample_image):
	numrows, numcols = sample_image.shape
	col = int(x+0.5)
	row = int(y+0.5)
	if col>= 0 and col<numcols and row>=0 and row<numrows:
		z = sample_image[row,col]
		return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x,y,z)
	else:
		return 'x=%1.4f, y=1.4%f'%(x,y)

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

def process_image(channel_img, win_x, win_y, std = False, remove_zeros = False):

	if std:
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
		std = np.std(channel_img)
		channel_img /= std
		return channel_img

	if remove_zeros:
		channel_img /= 255
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
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
		self.init = initializations.get(init)
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
		self.init = initializations.get(init)
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
	rotate = True, flip = True, shear = 0, class_weight = None):

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
						class_weight = class_weight,
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
		model_output = model_output[0,:,:,:]
		
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

def run_models_on_directory(data_location, channel_names, output_location, model_fn, list_of_weights, n_features = 3, image_size_x = 1080, image_size_y = 1280, win_x = 30, win_y = 30, std = False, split = True, process = True, save = True):

	batch_input_shape = (1,len(channel_names),image_size_x+win_x, image_size_y+win_y)
	model = model_fn(batch_input_shape = batch_input_shape, n_features = n_features, weights_path = list_of_weights[0])
	n_features = model.layers[-1].output_shape[1]

	model_outputs = []
	for weights_path in list_of_weights:
		model = set_weights(model, weights_path = weights_path)
		processed_image_list= run_model_on_directory(data_location, channel_names, output_location, model, win_x = win_x, win_y = win_y, save = False, std = std, split = split, process = process)
		model_outputs += [np.stack(processed_image_list, axis = 0)]

	# Average all images
	model_output = np.stack(model_outputs, axis = 0)
	model_output = np.mean(model_output, axis = 0)
		
	# Save images
	if save:
		for img in xrange(model_output.shape[0]):
			for feat in xrange(n_features):
				feature = model_output[img,feat,:,:]
				cnnout_name = os.path.join(output_location, 'feature_' + str(feat) + "_frame_" + str(img) + r'.tif')
				tiff.imsave(cnnout_name,feature)

	from keras.backend.common import _UID_PREFIXES
	for key in _UID_PREFIXES:
		_UID_PREFIXES[key] = 0

	return model_output

def run_model_on_lsm(lsm_file, output_location, model, win_x = 15, win_y = 15, std = False, split = True, save = True):
	n_features = model.layers[-1].output_shape[1]
	counter = 0

	from pylsm import lsmreader
	raw_image_file = lsmreader.Lsmimage(lsm_file)
	raw_image_file.open()

	image_size_x = raw_image_file.image['data'][0].shape[0]
	image_size_y = raw_image_file.image['data'][0].shape[1]
	image_size_z = raw_image_file.image['data'][0].shape[2]
	num_channels = len(raw_image_file.image['data'])

	channels = np.zeros((image_size_z, num_channels, image_size_x, image_size_y), dtype = 'float32')
	processed_image_list = []

	for zpos in xrange(image_size_z):
		for channel in xrange(num_channels):
			channel_img = raw_image_file.get_image(stack = zpos, channel = channel)
			channel_img = np.float32(channel_img)
			channel_img = process_image(channel_img, win_x, win_y, remove_zeros = True)
			channels[zpos, channel, :, :] = channel_img

	for zpos in xrange(image_size_z):
		print "Processing image " + str(counter + 1) + " of " + str(image_size_z)
		image = np.zeros((1, num_channels, image_size_x, image_size_y))
		image[0,:,:,:] = channels[zpos,:,:,:]
		processed_image = run_model(image, model, win_x = win_x, win_y = win_y, std = std, split = split, process = False)
		processed_image_list += [processed_image]

		# Save images
		if save:
			for feat in xrange(n_features):
				feature = processed_image[feat,:,:]
				cnnout_name = os.path.join(output_location, 'feature_' + str(feat) +"_frame_"+ str(counter) + r'.tif')
				tiff.imsave(cnnout_name,feature)
		counter += 1

	return processed_image_list

def run_models_on_lsm(lsm_file, output_location, model_fn, list_of_weights, n_features = 3, image_size_x = 512, image_size_y = 512, win_x = 15, win_y = 15, std = False, split = True, save = True):

	from pylsm import lsmreader
	raw_image_file = lsmreader.Lsmimage(lsm_file)
	raw_image_file.open()

	image_size_x = raw_image_file.image['data'][0].shape[0]
	image_size_y = raw_image_file.image['data'][0].shape[1]
	image_size_z = raw_image_file.image['data'][0].shape[2]
	num_channels = len(raw_image_file.image['data'])

	batch_input_shape = (1,num_channels,image_size_x+win_x, image_size_y+win_y)
	model = model_fn(batch_input_shape = batch_input_shape, n_features = n_features, weights_path = list_of_weights[0])
	n_features = model.layers[-1].output_shape[1]

	model_outputs = []
	for weights_path in list_of_weights:
		model = set_weights(model, weights_path = weights_path)
		processed_image_list= run_model_on_lsm(lsm_file, output_location, model, win_x = win_x, win_y = win_y, save = True, std = std, split = split)
		model_outputs += [np.stack(processed_image_list, axis = 0)]

	# Average all images
	model_output = np.stack(model_outputs, axis = 0)
	model_output = np.mean(model_output, axis = 0)

	# Save images
	if save:
		for img in xrange(model_output.shape[0]):
			for feat in xrange(n_features):
				feature = model_output[img,feat,:,:]
				cnnout_name = os.path.join(output_location, 'feature_' + str(feat) + "_frame_" + str(img) + r'.tif')
				tiff.imsave(cnnout_name,feature)

	from keras.backend.common import _UID_PREFIXES
	for key in _UID_PREFIXES:
		_UID_PREFIXES[key] = 0

	return model_output




"""
Active contours
"""

from itertools import cycle

import numpy as np
import scipy
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, \
						gaussian_filter, gaussian_gradient_magnitude

import mahotas as mh
from skimage.segmentation import find_boundaries
from skimage.measure import label
from skimage import morphology as morph

def zero_crossing(data, offset = 0.5):
	new_data = data - offset
	zc = np.where(np.diff(np.signbit(new_data)))[0]

	return zc

"""
Define active contour functions
"""
class fcycle(object):
	
	def __init__(self, iterable):
		"""Call functions from the iterable each time it is called."""
		self.funcs = cycle(iterable)
	
	def __call__(self, *args, **kwargs):
		f = self.funcs.next()
		return f(*args, **kwargs)
	

# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3), np.array([[0,1,0]]*3), np.flipud(np.eye(3)), np.rot90([[0,1,0]]*3)]
_P3 = [np.zeros((3,3,3)) for i in xrange(9)]

_P3[0][:,:,1] = 1
_P3[1][:,1,:] = 1
_P3[2][1,:,:] = 1
_P3[3][:,[0,1,2],[0,1,2]] = 1
_P3[4][:,[0,1,2],[2,1,0]] = 1
_P3[5][[0,1,2],:,[0,1,2]] = 1
_P3[6][[0,1,2],:,[2,1,0]] = 1
_P3[7][[0,1,2],[0,1,2],:] = 1
_P3[8][[0,1,2],[2,1,0],:] = 1

_aux = np.zeros((0))
def SI(u):
	"""SI operator."""
	global _aux
	if np.ndim(u) == 2:
		P = _P2
	elif np.ndim(u) == 3:
		P = _P3
	else:
		raise ValueError, "u has an invalid number of dimensions (should be 2 or 3)"
	
	if u.shape != _aux.shape[1:]:
		_aux = np.zeros((len(P),) + u.shape)
	
	for i in xrange(len(P)):
		_aux[i] = binary_erosion(u, P[i])
	
	return _aux.max(0)

def IS(u):
	"""IS operator."""
	global _aux
	if np.ndim(u) == 2:
		P = _P2
	elif np.ndim(u) == 3:
		P = _P3
	else:
		raise ValueError, "u has an invalid number of dimensions (should be 2 or 3)"
	
	if u.shape != _aux.shape[1:]:
		_aux = np.zeros((len(P),) + u.shape)
	
	for i in xrange(len(P)):
		_aux[i] = binary_dilation(u, P[i])
	
	return _aux.min(0)

# SIoIS operator.
SIoIS = lambda u: SI(IS(u))
ISoSI = lambda u: IS(SI(u))
curvop = fcycle([SIoIS, ISoSI])


class MorphACWE(object):
	"""Morphological ACWE based on the Chan-Vese energy functional."""
	
	def __init__(self, data, smoothing=1, lambda1=10, lambda2=1):
		"""Create a Morphological ACWE solver.
		
		Parameters
		----------
		data : ndarray
			The image data.
		smoothing : scalar
			The number of repetitions of the smoothing step (the
			curv operator) in each iteration. In other terms,
			this is the strength of the smoothing. This is the
			parameter mu.
		lambda1, lambda2 : scalars
			Relative importance of the inside pixels (lambda1)
			against the outside pixels (lambda2).
		"""

		self._u = None
		self.smoothing = smoothing
		self.lambda1 = lambda1
		self.lambda2 = lambda2
		
		self.data = data
		self.mask = data
	
	def set_levelset(self, u):
		self._u = np.double(u)
		self._u[u>0] = 1
		self._u[u<=0] = 0
	
	levelset = property(lambda self: self._u,
						set_levelset,
						doc="The level set embedding function (u).")
	
	def step(self):
		"""Perform a single step of the morphological Chan-Vese evolution."""
		# Assign attributes to local variables for convenience.
		u = self._u
		
		if u is None:
			raise ValueError, "the levelset function is not set (use set_levelset)"
		
		data = self.data
		
		# Create mask to separate objects
		labeled, nr_objects = mh.label(u)
		mask = mh.segmentation.gvoronoi(labeled)
		mask = 1-find_boundaries(mask)

		self.mask = np.float32(mask)/np.float32(mask).max()

		# Determine c0 and c1.
		inside = u>0
		outside = u<=0
		c0 = data[outside].sum() / float(outside.sum())
		c1 = data[inside].sum() / float(inside.sum())
		
		# Image attachment.
		dres = np.array(np.gradient(u))
		abs_dres = np.abs(dres).sum(0)
		aux = abs_dres * (self.lambda1*(data - c1)**2 - self.lambda2*(data - c0)**2)
		
		res = np.copy(u)
		res[aux < 0] = 1
		res[aux > 0] = 0
		
		# Smoothing.
		for i in xrange(self.smoothing):
			res = curvop(res)
		
		# Apply mask
		res *= mask

		self._u = res
	
	def run(self, iterations):
		"""Run several iterations of the morphological Chan-Vese method."""
		for i in xrange(iterations):
			self.step()
	
def segment_image_w_morphsnakes(img, nuc_label, num_iters, smoothing = 2):
	morph_snake = MorphACWE(img, smoothing = smoothing, lambda1 = 1, lambda2 = 1)
	morph_snake.levelset = np.float16(nuc_label > 0)

	for j in xrange(num_iters):
		morph_snake.step()

	seg_input = morph_snake.levelset
	seg = morph.watershed(seg_input, nuc_label, mask = (seg_input > 0))
	return seg

"""
Helper functions for segmentation
"""

def segment_nuclei(img = None, save = True, adaptive = False, color_image = False, load_from_direc = None, feature_to_load = "feature_1", mask_location = None, threshold = 0.5, area_threshold = 50, eccentricity_threshold = 1, solidity_threshold = 0):
	# Requires a 4 channel image (number of frames, number of features, image width, image height)
	from skimage.filters import threshold_otsu, threshold_adaptive

	if load_from_direc is None:
		img = img[:,1,:,:]
		nuclear_masks = np.zeros(img.shape, dtype = np.float32)

	if load_from_direc is not None:
		img_files = nikon_getfiles(load_from_direc, feature_to_load )
		img_size = get_image_sizes(load_from_direc, feature_to_load)
		img = np.zeros((len(img_files), img_size[0], img_size[1]), dtype = np.float32)
		nuclear_masks = np.zeros((len(img_files), img.shape[1], img.shape[2]), dtype = np.float32)
		counter = 0
		for name in img_files:
			img[counter,:,:] = get_image(os.path.join(load_from_direc,name))
			counter += 1

	for frame in xrange(img.shape[0]):
		interior = img[frame,:,:]
		if adaptive:
			block_size = 61
			nuclear_mask = np.float32(threshold_adaptive(interior, block_size, method = 'median', offset = -.075))
		else: 
			nuclear_mask = np.float32(interior > threshold)
		nuc_label = label(nuclear_mask)
		max_cell_id = np.amax(nuc_label)
		for cell_id in xrange(1,max_cell_id + 1):
			img_new = nuc_label == cell_id
			img_fill = binary_fill_holes(img_new)
			nuc_label[img_fill == 1] = cell_id

		region_temp = regionprops(nuc_label)

		for region in region_temp:
			if region.area < area_threshold:
				nuclear_mask[nuc_label == region.label] = 0
			if region.eccentricity > eccentricity_threshold:
				nuclear_mask[nuc_label == region.label] = 0
			if region.solidity < solidity_threshold:
				nuclear_mask[nuc_label == region.label] = 0

		nuclear_masks[frame,:,:] = nuclear_mask

		if save:
			img_name = os.path.join(mask_location, "nuclear_mask_" + str(frame) + ".png")
			tiff.imsave(img_name,nuclear_mask)

		if color_image:
			img_name = os.path.join(mask_location, "nuclear_colorimg_" + str(frame) + ".png")
			
			from skimage.segmentation import find_boundaries
			import palettable
			from skimage.color import label2rgb

			seg = label(nuclear_mask)
			bound = find_boundaries(seg, background = 0)

			image_label_overlay = label2rgb(seg, bg_label = 0, bg_color = (0.8,0.8,0.8), colors = palettable.colorbrewer.sequential.YlGn_9.mpl_colors)
			image_label_overlay[bound == 1,:] = 0

			scipy.misc.imsave(img_name,np.float32(image_label_overlay))
	return nuclear_masks

def segment_cytoplasm(img =None, save = True, load_from_direc = None, feature_to_load = "feature_1", color_image = False, nuclear_masks = None, mask_location = None, smoothing = 1, num_iters = 80):
	if load_from_direc is None:
		cytoplasm_masks = np.zeros((img.shape[0], img.shape[2], img.shape[3]), dtype = np.float32)
		img = img[:,1,:,:]

	if load_from_direc is not None:
		img_files = nikon_getfiles(load_from_direc, feature_to_load )
		img_size = get_image_sizes(load_from_direc, feature_to_load)
		img = np.zeros((len(img_files), img_size[0], img_size[1]), dtype = np.float32)
		cytoplasm_masks = np.zeros((len(img_files), img.shape[1], img.shape[2]), dtype = np.float32)

		counter = 0
		for name in img_files:
			img[counter,:,:] = get_image(os.path.join(load_from_direc,name))
			counter += 1

	for frame in xrange(img.shape[0]):
		interior = img[frame,:,:]

		nuclei = nuclear_masks[frame,:,:]

		nuclei_label = label(nuclei, background = 0)

		seg = segment_image_w_morphsnakes(interior, nuclei_label, num_iters = num_iters, smoothing = smoothing)
		seg[seg == 0] = -1

		cytoplasm_mask = np.zeros(seg.shape,dtype = np.float32)
		max_cell_id = np.amax(seg)
		for cell_id in xrange(1,max_cell_id + 1):
			img_new = seg == cell_id
			img_fill = binary_fill_holes(img_new)
			cytoplasm_mask[img_fill == 1] = 1

		cytoplasm_masks[frame,:,:] = cytoplasm_mask

		if save:
			img_name = os.path.join(mask_location, "cytoplasm_mask_" + str(frame) + ".png")
			tiff.imsave(img_name,np.float32(cytoplasm_mask))

		if color_image:
			img_name = os.path.join(mask_location, "cytoplasm_colorimg_" + str(frame) + ".png")
			
			from skimage.segmentation import find_boundaries
			import palettable
			from skimage.color import label2rgb

			seg = label(cytoplasm_mask)
			bound = find_boundaries(seg, background = 0)

			image_label_overlay = label2rgb(seg, bg_label = 0, bg_color = (0.8,0.8,0.8), colors = palettable.colorbrewer.sequential.YlGn_9.mpl_colors)
			image_label_overlay[bound == 1,:] = 0

			scipy.misc.imsave(img_name,np.float32(image_label_overlay))

	return cytoplasm_masks

"""
Helper functions for jaccard and dice indices
"""

def dice_jaccard_indices(mask, val, nuc_mask):

	strel = morph.disk(1)
	val = morph.erosion(val,strel)
	mask = mask.astype('int16')
	val = val.astype('int16')

	mask_label = label(mask, background = 0) 
	val_label = label(val, background = 0) 

	for j in xrange(1,np.amax(val_label)+1):
		if np.sum((val_label == j) * nuc_mask) == 0:
			val_label[val_label == j] = 0

	val_label = label(val_label > 0, background = 0) 

	mask_region = regionprops(mask_label)
	val_region = regionprops(val_label)

	jac_list = []
	dice_list = []

	for val_prop in val_region:
		temp = val_prop['coords'].tolist()
		internal_points_1 = set([tuple(l) for l in temp])
		best_mask_prop = mask_region[0]
		best_overlap = 0
		best_sum = 0
		best_union = 0

		for mask_prop in mask_region:
			temp = mask_prop['coords'].tolist()
			internal_points_2 = set([tuple(l) for l in temp])

			overlap = internal_points_1 & internal_points_2
			num_overlap = len(overlap)

			if num_overlap > best_overlap:
				best_mask_prop = mask_prop
				best_overlap = num_overlap
				best_union = len(internal_points_1 | internal_points_2)
				best_sum = len(internal_points_1) + len(internal_points_2)

		jac = np.float32(best_overlap)/np.float32(best_union)
		dice = np.float32(best_overlap)*2/best_sum

		if np.isnan(jac) == 0 and np.isnan(dice) == 0:
			jac_list += [jac]
			dice_list += [dice]

	JI = np.mean(jac_list)
	DI = np.mean(dice_list)
	print jac_list, dice_list
	print "Jaccard index is " + str(JI) + " +/- " + str(np.std(jac_list))
	print "Dice index is " + str(DI)  + " +/- " + str(np.std(dice_list))

	return JI, DI

"""
Functions for tracking bacterial cells from frame to frame
"""

def create_masks(direc_name, direc_save_mask, direc_save_region, win = 15, area_threshold = 30, eccen_threshold = 0.1, clear_borders = 0):

	imglist_int = nikon_getfiles(direc_name,'feature_1')
	imglist_back = nikon_getfiles(direc_name,'feature_2')
	imglist_bound = nikon_getfiles(direc_name,'feature_0')
	num_of_files = len(imglist_int)

	# Create masks of chunks
	iterations = 0
	cnn_int_name = os.path.join(direc_name, imglist_int[iterations])
	mask_interior = get_image(cnn_int_name)[win:-win,win:-win]
	mask_sum = np.zeros(mask_interior.shape)
	mask_save = np.zeros((num_of_files,mask_interior.shape[0],mask_interior.shape[1]))

	for iterations in xrange(num_of_files):

		cnn_int_name = os.path.join(direc_name, imglist_int[iterations])
		cnn_back_name = os.path.join(direc_name, imglist_back[iterations])
		cnn_bound_name = os.path.join(direc_name, imglist_bound[iterations])

		mask_interior = get_image(cnn_int_name)[win:-win,win:-win]
		mask_background = get_image(cnn_back_name)[win:-win,win:-win]
		mask_boundary = get_image(cnn_bound_name)[win:-win,win:-win]

		thresh = threshold_otsu(mask_interior)
		mask_interior_thresh = np.float32(mask_interior>0.6)

		# Screen cell size
		mask_interior_label = label(mask_interior_thresh)
		region_temp = regionprops(mask_interior_label)
		for region in region_temp:
			if region.area < area_threshold:
				mask_interior_thresh[mask_interior_label == region.label] = 0
			if region.eccentricity < eccen_threshold:
				mask_interior_thresh[mask_interior_label == region.label] = 0

		# Clear borders
		if clear_borders == 1:
			mask_interior_thresh = np.float32(clear_border(mask_interior_thresh))

		mask_save[iterations,:,:] = np.float32(mask_interior_thresh)


		# Save thresholded masks
		print '... Saving mask number ' + str(iterations+1) + ' of ' + str(len(imglist_int)) + '\r',
		file_name_save = 'masks_' + str(iterations) + '.tif'
		tiff.imsave(direc_save_mask + file_name_save, mask_interior_thresh)

		mask_sum += mask_interior_thresh

	mask_interior_thresh = mask_sum > 0
	strel = morph.disk(3)
	mask_closed	= morph.binary_closing(mask_interior_thresh,strel)
	mask_holes_filled = binary_fill_holes(mask_closed)
	markers = sk.measure.label(mask_holes_filled)
	markers = np.asarray(markers,dtype = np.int32)

	file_name_save = 'chunk_markers' + '.tif'
	tiff.imsave(direc_save_mask + file_name_save, markers)

	num_of_chunks = np.amax(markers) + 1

	fig,ax = plt.subplots(1,1)
	ax.imshow(markers,cmap=plt.cm.gray,interpolation='nearest')
	def f_coord(x,y):
		return form_coord(x,y,markers)
	ax.format_coord = f_coord

	plt.show()

	regions_save = []

	for chunk in xrange(num_of_chunks):
		regions = []
		chunk_mask = markers == chunk

		for iterations in xrange(num_of_files):
			mask_interior_thresh = mask_save[iterations,:,:] * chunk_mask
			mask_interior_label = label(mask_interior_thresh, background = 0)
			if chunk == 5:
				file_name_save = 'chunk_5_' + str(iterations) + '.tif'
				tiff.imsave(direc_save_mask + file_name_save, np.float32(mask_interior_label))
			# Obtain region properties
			regions.append(regionprops(mask_interior_label))

		regions_save.append(regions)

	# Save region properties
	file_name_save = 'regions_save.npz'
	np.savez(os.path.join(direc_save_region,file_name_save), regions_save=regions_save)

	return None

def crop_images(direc_name, channel_names, direc_save, window_size_x = 15, window_size_y = 15):
	imglist = []
	for j in xrange(len(channel_names)):
		imglist.append(nikon_getfiles(direc_name,channel_names[j]))
	
	for i in xrange(len(imglist)):
		for j in xrange(len(imglist[0])):
			im = get_image(direc_name + imglist[i][j])
			im_crop = im[window_size_x:-window_size_x,window_size_y:-window_size_y]

			tiff.imsave(direc_save + imglist[i][j], im_crop)

def align_images(direc_name, channel_names, direc_save,crop_window = 950):
	# Make sure the first member of channel name is the phase image
	imglist = []
	for j in xrange(len(channel_names)):
		imglist.append(nikon_getfiles(direc_name,channel_names[j]))

	for j in xrange(len(imglist[0])-1):
		im0_name = os.path.join(direc_name, imglist[0][j])
		im1_name = os.path.join(direc_name, imglist[0][j+1])

		im0 = get_image(im0_name)
		im1 = get_image(im1_name)

		image_size_x = im0.shape[0]
		image_size_y = im0.shape[1]

		x_index = np.floor(image_size_x/2) - crop_window/2 - 1 
		y_index = np.floor(image_size_y/2) - crop_window/2 - 1

		im0_crop = im0[x_index : x_index + crop_window, y_index : y_index + crop_window]
		im1_crop = im1[x_index : x_index + crop_window, y_index : y_index + crop_window]

		shape = im0_crop.shape
		f0 = fft2(im0_crop)
		f1 = fft2(im1_crop)
		ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
		t0, t1 = np.unravel_index(np.argmax(ir), shape)
		if t0 > shape[0] // 2:
			t0 -= shape[0]
		if t1 > shape[1] // 2:
			t1 -= shape[1]

		im0_save = im0_crop[25:-25,25:-25]
		im1_save = im1_crop[25 - t0: -25 - t0, 25 - t1: -25 -t1]

		if j == 0:
			im_name = os.path.join(direc_save, channel_names[0] +'_aligned_' + str(j) + '.tif')
			tiff.imsave(im_name , im0_save)
			# Load, shift, and save fluorescence channels
			for i in xrange(1,len(channel_names)):
				im = get_image(direc_name + imglist[i][j])
				im_crop = im[x_index : x_index + crop_window, y_index : y_index + crop_window]
				im_save = im_crop[25:-25,25:-25]
				im_name = os.path.join(direc_save, channel_names[i] +'_aligned_' + str(j) + '.tif')
				tiff.imsave(im_name, im_save)
			# print '... Aligned frame ' + str(j+1) + ' of ' + str(len(imglist[0])) + '\r',

		tiff.imsave(direc_save + channel_names[0] + '_aligned_' + str(j+1) + '.tif', im1_save)
		
		# Load, shift, and save fluorescence channels
		for i in xrange(1,len(channel_names)):
			im = get_image(direc_name + imglist[i][j+1])
			im_crop = im[x_index : x_index + crop_window, y_index : y_index + crop_window]
			im_save = im_crop[25 - t0:-25 - t0, 25 - t1:-25 - t1]
			im_name = os.path.join(direc_save, channel_names[i] +'_aligned_' + str(j) + '.tif')

			tiff.imsave(im_name, im_save)

		# print '... Aligned frame ' + str(j+2) + ' of ' + str(len(imglist[0])) + '\r',
	return None

def make_cost_matrix(region_1, region_2, frame_numbers, direc_save, birth_cost = 1e6, death_cost = 1e5, no_division_cost = 100):
	N_1 = len(region_1)
	N_2 = len(region_2)
	cost_matrix = np.zeros((2*N_1+N_2,2*N_1+N_2), dtype = np.double)
	for i in xrange(N_1):
		print '... Processing ' + str(np.floor(np.float32(i)/np.float32(N_1)*100)) + r'% complete with image' + '\r',
		for j in xrange(N_2):
			cost_matrix[i,j] = cost_function_overlap_daughter(region_1[i],region_2[j])

	cost_matrix[N_1:2*N_1,0:N_2] = cost_matrix[0:N_1,0:N_2]

	for i in xrange(N_1):
		print '... Processing ' + str(np.floor(np.float32(i)/np.float32(N_1)*100)) + r'% complete with image' + '\r',
		for j in xrange(N_2):
			cost_matrix[i,j] = cost_function_overlap(region_1[i],region_2[j])

	births = np.eye(N_2,N_2) * birth_cost
	births[births == 0] = np.Inf
	cost_matrix[2*N_1:,0:N_2] = births

	deaths = np.eye(N_1,N_1) * death_cost
	deaths[deaths == 0] = np.Inf
	cost_matrix[0:N_1,N_2:N_2+N_1] = deaths

	no_division = np.eye(N_1,N_1) * no_division_cost
	no_division[no_division == 0] = np.Inf
	cost_matrix[N_1:2*N_1,N_1+N_2:] = no_division

	cost_matrix[N_1:2*N_1,N_2:N_2+1] = np.Inf*np.ones(cost_matrix[N_1:2*N_1,N_2:N_2+1].shape)
	cost_matrix[0:N_1,N_1+N_2:] = np.Inf*np.ones(cost_matrix[0:N_1,N_1+N_2:].shape)

	cost_matrix[2*N_1:,N_2:] = .001*cost_matrix[0:2*N_1,0:N_2].T

	frame_1 = str(frame_numbers[0])
	frame_2 = str(frame_numbers[1])
	file_name_save = direc_save + 'cost_matrix_' + frame_1 + '_' + frame_2
	np.savez(file_name_save, cost_matrix)

	return cost_matrix

def cost_function_centroid(cell1, cell2, max_dist = 20):
	centroid_1 = np.asarray(cell1.centroid)
	centroid_2 = np.asarray(cell2.centroid)

	temp = np.sum((centroid_1-centroid_2) ** 2)
	if np.sqrt(temp > max_dist):
		temp = np.Inf
	return temp

def cost_function_overlap(cell1,cell2, max_dist = 15):
	internal_points_1 = cell1['coords'].tolist()
	internal_points_2 = cell2['coords'].tolist()

	ip_1 = set([str(x) for x in internal_points_1])
	ip_2 = set([str(x) for x in internal_points_2])

	area_1 = cell1['area']
	area_2 = cell2['area']

	centroid_1 = np.floor(np.asarray(cell1['centroid']))
	centroid_2 = np.floor(np.asarray(cell2['centroid']))

	dist = np.sqrt(np.sum((centroid_1-centroid_2) ** 2))

	if dist < max_dist:
		counter_1 = 0
		counter_2 = 0
		for point in internal_points_1:
			centroid = np.round(centroid_2)
			if centroid[0] == point[0] and centroid[1] == point[1]:
				counter_1 += 1

		for point in internal_points_2:
			centroid = np.round(centroid_1)
			if centroid[0] == point[0] and centroid[1] == point[1]:
				counter_2 += 1

		if counter_1 > 0 or counter_2 > 0:
			cost = -1e6
		else:

			overlap = [point for point in internal_points_1 if point in internal_points_2]
			num_overlap = len(overlap)

			frac_overlap = np.amin([num_overlap/area_1, num_overlap/area_2])
			if frac_overlap > 0.1:
				cost = -1e5*frac_overlap

			if area_1 > 3* area_2 or area_2 > 3*area_1:
				cost = 1e6
			
			else: 
				cost = 1e4
	else:
		cost = np.Inf

	return cost

def cost_function_overlap_daughter(cell1,cell2, max_dist = 25):
	internal_points_1 = cell1['coords'].tolist()
	internal_points_2 = cell2['coords'].tolist()

	area_1 = cell1['area']
	area_2 = cell2['area']

	centroid_1 = np.floor(np.asarray(cell1['centroid']))
	centroid_2 = np.floor(np.asarray(cell2['centroid']))

	dist = np.sqrt(np.sum((centroid_1-centroid_2) ** 2))

	if dist < max_dist:
		
		counter_1 = 0
		counter_2 = 0
		for point in internal_points_1:
			centroid = np.round(centroid_2)
			if centroid[0] == point[0] and centroid[1] == point[1]:
				counter_1 += 1

		if counter_1 > 0:
			cost = -10000

		else:
			vector_1 = centroid_1-centroid_2
			vector_2 = centroid_2 + cell2['major_axis_length']*np.array([np.sin(cell2['orientation']),np.cos(cell2['orientation']),])

			cosangle = np.dot(vector_1,vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))

			overlap = [point for point in internal_points_1 if point in internal_points_2]
			num_overlap = len(overlap)

			frac_overlap = np.amin([num_overlap/area_1, num_overlap/area_2])
			if frac_overlap > 0.1:
				cost = -1000*frac_overlap

			if area_1 > 5* area_2 or area_2 > 5* area_1:
				cost = 1e6

			else: 
				cost = 1e6

	else:
		cost = np.Inf

	return cost

def run_LAP(cost_matrix, N_1, N_2):
	from scipy.optimize import linear_sum_assignment as scipy_lap
	assignment = scipy_lap(cost_matrix)

	x = assignment[0]
	y = assignment[1]

	binaryAssign = np.zeros(cost_matrix.shape, bool)
	binaryAssign[x, y] = True

	binaryAssign = binaryAssign[0:2*N_1, 0:N_2]

	idx, idy = np.where(binaryAssign)
	return idx + 1, idy + 1

def cell(prop, frame):
	cell = {}
	cell['area'] = prop['area']
	cell['xcentroid'] = prop['centroid'][1]
	cell['ycentroid'] = prop['centroid'][0]
	cell['coords'] = prop['coords']
	cell['length'] = prop['major_axis_length']
	cell['width'] = prop['minor_axis_length']
	cell['orientation'] = prop['orientation']
	cell['cellId'] = prop['label']
	cell['trackId'] = np.nan # once a cell has been assigned to a track, change this to the track's number
	cell['tracked'] = np.nan  # once there is an object, change this into 0
	cell['parentId'] = np.nan
	cell['frame'] = frame
	return cell

def cell_linker_init(region_1, frame):
	# This function intializes the tracks for the first image
	tracks = []
	for prop in region_1:
		cell_to_add = cell(prop,frame)
		trackId = len(tracks)
		cell_to_add['trackId'] = trackId
		cell_to_add['tracked'] = 0
		tracks.append([cell_to_add])
	return tracks

def cell_linker(region_1, region_2, tracks, frame_numbers, direc_save):

	# Find what tracks the cells in image 1 belong to
	track_location = {}
	for track in tracks:
		if track[-1]['frame'] == frame_numbers[0]:
			track_location[str(track[-1]['cellId'])] = track[-1]['trackId']

	# Create cells for all of the cells in image 2
	image_2_cells = []
	for prop in region_2:
		image_2_cells.append(cell(prop,frame_numbers[1]))

	# Create cost matrix for LAP problem
	cost_matrix = make_cost_matrix(region_1,region_2, frame_numbers, direc_save)
	N_1 = len(region_1)
	N_2 = len(region_2)

	# Run LAP
	assigned_1, assigned_2 = run_LAP(cost_matrix, N_1, N_2)

	# Add assigned cells to tracks
	for j in xrange(len(assigned_2)):
		if assigned_1[j] < N_1 + 1:
			image_2_cells[assigned_2[j]-1]['trackId'] = track_location[str(assigned_1[j])]
			image_2_cells[assigned_2[j]-1]['tracked'] = 0
			image_2_cells[assigned_2[j]-1]['parentId'] = tracks[track_location[str(assigned_1[j])]][-1]['parentId']
			tracks[track_location[str(assigned_1[j])]].append(image_2_cells[assigned_2[j]-1])
		else:
			# DOUBLE CHECK: If a daughter cell is assigned, make sure the original cell wasn't assigned to cell death - if so set daughter cell to be the original cell
			orig_cell_id = assigned_1[j]-N_1
			check_assignment = np.sum(assigned_1 == orig_cell_id)
			if check_assignment == 1:
				image_2_cells[assigned_2[j]-1]['parentId'] = track_location[str(assigned_1[j]-N_1)]
			else:
				image_2_cells[assigned_2[j]-1]['trackId'] = track_location[str(orig_cell_id)]
				image_2_cells[assigned_2[j]-1]['tracked'] = 0
				image_2_cells[assigned_2[j]-1]['parentId'] = tracks[track_location[str(orig_cell_id)]][-1]['parentId']
				tracks[track_location[str(orig_cell_id)]].append(image_2_cells[assigned_2[j]-1])

	# Re scan for untracked cells
	untracked = []
	for im2_cell in image_2_cells:
		if np.isnan(im2_cell['tracked']) == 1:
			untracked.append(im2_cell)

	# Create new tracks containing the new cells
	num_of_tracks = len(tracks)
	counter = 0
	for im2_cell in untracked:
		im2_cell['trackId'] = num_of_tracks + counter
		im2_cell['tracked'] = 0
		counter += 1
		tracks.append([im2_cell])

	# Return the list of tracks
	return tracks

def make_tracks(regions, direc_save, start_frame = 0, end_frame = None, direc_cost_save = None):
	if end_frame == None:
		end_frame = len(regions)
	tracks = cell_linker_init(regions[start_frame],start_frame)

	for j in xrange(start_frame,end_frame-1):
		tracks = cell_linker(regions[j],regions[j+1],tracks, frame_numbers = [j, j+1], direc_save = direc_cost_save)
		print '... Tracked image ' + str(j) + '...' + str(len(tracks)) + ' tracks identified'

	file_name_save = 'tracks'
	np.savez(direc_save + file_name_save, tracks = tracks)

	return tracks

def get_lineage(tracks,trackID):
	list_of_cells = []
	lineage_ids = [trackID]
	list_of_cells = tracks[trackID]

	# Find daughter cells
	for lineage_id in lineage_ids:
		for track in tracks:
			if track[0]['parentId'] == lineage_id:
				lineage_ids.append(track[0]['trackId'])
				list_of_cells += track

	return list_of_cells, lineage_ids

def plot_lineage(list_of_cells, tracks, image_size):

	# Construct a mask with all the cells
	all_cells = []
	for track in tracks:
		all_cells += track

	# Find out the number of frames
	min_frame_number = np.Inf
	max_frame_number = -np.Inf

	for cell in list_of_cells:
		if cell['frame'] < min_frame_number:
			min_frame_number = cell['frame']
		if cell['frame'] > max_frame_number:
			max_frame_number = cell['frame']

	num_of_frames = max_frame_number - min_frame_number + 1
	print max_frame_number, min_frame_number

	# Create array with masks of each for each frame
	mask_array = np.zeros((num_of_frames,image_size[0],image_size[1]))
	all_cell_mask = np.zeros((num_of_frames,image_size[0],image_size[1]))

	for cell in all_cells:
		coords_x = cell['coords'][:,0]
		coords_y = cell['coords'][:,1]
		frame_id = cell['frame']
		if frame_id > min_frame_number-1 and frame_id < max_frame_number+1:
			all_cell_mask[frame_id-min_frame_number,coords_x,coords_y] = 1

	for cell in list_of_cells:
		coords_x = cell['coords'][:,0]
		coords_y = cell['coords'][:,1]
		frame_id = cell['frame']

		mask_array[frame_id-min_frame_number,coords_x,coords_y] = 1

	# Find a bounding box for all of the coordinates
	mask_all_cells = np.sum(mask_array, axis=0)
	bound = np.argwhere(mask_all_cells)
	(row_start, col_start), (row_stop, col_stop) = bound.min(0), bound.max(0) + 1
	mask_array_trim = mask_array[:,row_start:row_stop,col_start:col_stop]
	all_cell_trim = all_cell_mask[:,row_start:row_stop,col_start:col_stop]

	# Display arrays - convert each mask to rgb label

	fig,ax = plt.subplots(2,num_of_frames, squeeze = False)

	for frame_number in xrange(num_of_frames):
		mask = mask_array_trim[frame_number,:,:]
		all_cell_image = all_cell_trim[frame_number,:,:]
		label_image = label(mask)
		image_label_overlay = label2rgb(label_image,mask)

		ax[0,frame_number].imshow(image_label_overlay, interpolation = 'nearest')
		def form_coord(x,y):
			return cf.format_coord(x,y,label_image[:,:])
		ax[0,frame_number].format_coord = form_coord
		ax[0,frame_number].axes.get_xaxis().set_visible(False)
		ax[0,frame_number].axes.get_yaxis().set_visible(False)
		ax[0,frame_number].set_title(str(frame_number))

		ax[1,frame_number].imshow(all_cell_image, interpolation = 'nearest')
		def form_coord(x,y):
			return cf.format_coord(x,y,all_cell_image[:,:])
		ax[1,frame_number].format_coord = form_coord
		ax[1,frame_number].axes.get_xaxis().set_visible(False)
		ax[1,frame_number].axes.get_yaxis().set_visible(False)

	plt.show()

def plot_lineage_numbers(list_of_cells, tracks, image_size):

	# Construct a mask with all the cells
	all_cells = []
	for track in tracks:
		all_cells += track

	# Find out the number of frames
	min_frame_number = np.Inf
	max_frame_number = -np.Inf

	for cell in list_of_cells:
		if cell['frame'] < min_frame_number:
			min_frame_number = cell['frame']
		if cell['frame'] > max_frame_number:
			max_frame_number = cell['frame']

	num_of_frames = max_frame_number - min_frame_number + 1
	print max_frame_number, min_frame_number

	if num_of_frames > 10:

		# Create array with masks of each for each frame
		mask_array = np.zeros((num_of_frames,image_size[0],image_size[1]))
		all_cell_mask = np.zeros((num_of_frames,image_size[0],image_size[1]))

		for cell in all_cells:
			coords_x = cell['coords'][:,0]
			coords_y = cell['coords'][:,1]
			frame_id = cell['frame']
			if frame_id > min_frame_number-1 and frame_id < max_frame_number+1:
				all_cell_mask[frame_id-min_frame_number,coords_x,coords_y] = 1

		for cell in list_of_cells:
			coords_x = cell['coords'][:,0]
			coords_y = cell['coords'][:,1]
			frame_id = cell['frame']

			mask_array[frame_id-min_frame_number,coords_x,coords_y] = cell['trackId'] + 1

		# Find a bounding box for all of the coordinates
		mask_all_cells = np.sum(mask_array, axis=0)
		bound = np.argwhere(mask_all_cells)
		(row_start, col_start), (row_stop, col_stop) = bound.min(0), bound.max(0) + 1
		mask_array_trim = mask_array[:,row_start:row_stop,col_start:col_stop]
		all_cell_trim = all_cell_mask[:,row_start:row_stop,col_start:col_stop]

		# Display arrays - convert each mask to rgb label

		fig,ax = plt.subplots(2,num_of_frames, squeeze = False)
		form_coord_funcs = []
		form_coord_list = [0]*num_of_frames

		for frame_number in xrange(num_of_frames):
			mask = mask_array_trim[frame_number,:,:]
			all_cell_image = all_cell_trim[frame_number,:,:]

			ax[0,frame_number].imshow(mask, cmap = plt.cm.gist_stern, interpolation = 'nearest', vmin = 0, vmax = 50)
			# def form_coord(x,y):
			#     return cf.format_coord(x,y,mask_array_trim[frame_number,:,:])
			ax[0,frame_number].format_coord = lambda x,y: cf.format_coord(x,y,mask)
			ax[0,frame_number].axes.get_xaxis().set_visible(False)
			ax[0,frame_number].axes.get_yaxis().set_visible(False)
			ax[0,frame_number].set_title(str(frame_number))

			ax[1,frame_number].imshow(all_cell_image, cmap = plt.cm.gray,  interpolation = 'nearest')
			def form_coord(x,y):
				return cf.format_coord(x,y,all_cell_image[:,:])
			ax[1,frame_number].format_coord = form_coord
			ax[1,frame_number].axes.get_xaxis().set_visible(False)
			ax[1,frame_number].axes.get_yaxis().set_visible(False)
		plt.show()

def plot_lineage_total(list_of_cells, tracks, image_size):

	# Construct a mask with all the cells
	all_cells = []
	for track in tracks:
		all_cells += track

	# Find out the number of frames
	min_frame_number = np.Inf
	max_frame_number = -np.Inf

	for cell in list_of_cells:
		if cell['frame'] < min_frame_number:
			min_frame_number = cell['frame']
		if cell['frame'] > max_frame_number:
			max_frame_number = cell['frame']

	num_of_frames = max_frame_number - min_frame_number + 1
	print max_frame_number, min_frame_number

	# Create array with masks of each for each frame
	mask_array = np.zeros((total_no_of_frames,image_size[0],image_size[1]))
	all_cell_mask = np.zeros((total_no_of_frames,image_size[0],image_size[1]))

	# if num_of_frames == total_no_of_frames:

	for cell in all_cells:
		coords_x = cell['coords'][:,0]
		coords_y = cell['coords'][:,1]
		frame_id = cell['frame']
		if frame_id > min_frame_number-1 and frame_id < max_frame_number+1:
			all_cell_mask[frame_id-min_frame_number,coords_x,coords_y] = 1

	for cell in list_of_cells:
		coords_x = cell['coords'][:,0]
		coords_y = cell['coords'][:,1]
		frame_id = cell['frame']

		mask_array[frame_id-min_frame_number,coords_x,coords_y] = cell['trackId'] + 1

	return mask_array, all_cell_mask

''' For residual networks '''
def residual_block(block_function, n_filters, kernel, reps):
	def f(input):
		for i in range(reps):
			input = block_function(n_filters = n_filters, kernel=kernel)(input)
		return input

	return f
