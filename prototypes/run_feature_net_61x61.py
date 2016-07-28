'''Run a simple deep CNN on images.
GPU run command:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_feature_net_61x61.py

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
from keras import backend as K
import h5py
import tifffile as tiff

from cnn_functions import load_training_data, nikon_getfiles, get_image, process_image
from model_zoo import sparse_bn_feature_net_61x61, bn_feature_net_61x61

import os
import datetime
import numpy as np
import theano

"""
Load data
"""
direc_name = '/media/vanvalen/fe0ceb60-f921-4184-a484-b7de12c1eea6/Data/10.13.2015/run_1/run_conv/Pos15_5312016/'
image_dir = direc_name + 'RawImages/'
align_dir = direc_name + 'Align/'
cnn_save_dir = direc_name + 'Output/'
nuclei_dir = direc_name + 'Nuclei/'
mask_dir = direc_name + 'Masks/'
cropped_dir = direc_name + 'Cropped/'
channel_names = ['Phase','Far-red']
n_channels = len(channel_names)

win_x = 30
win_y = 30

data_location = os.path.join(direc_name, image_dir)
img_list_channels = []
for channel in channel_names:
	img_list_channels += [nikon_getfiles(data_location, channel)]
img_temp = get_image(data_location + img_list_channels[0][0])

image_size_x = img_temp.shape[0]/2
image_size_y = img_temp.shape[1]/2

combined_image = np.zeros((1,n_channels,img_temp.shape[0],img_temp.shape[1]), dtype = 'float32')
model_output = np.zeros((1,3,2*image_size_x-win_x*2, 2*image_size_y-win_y*2), dtype = 'float32')

stack_iteration = 0
for j in xrange(n_channels):
	channel_img = get_image(os.path.join(data_location, img_list_channels[j][stack_iteration]))
	print os.path.join(data_location, img_list_channels[j][stack_iteration])
	combined_image[0,j,:,:] = process_image(channel_img, 30, 30)

img_0 = combined_image[:,:, 0:image_size_x+win_x, 0:image_size_y+win_y]
img_1 = combined_image[:,:, 0:image_size_x+win_x, image_size_y-win_y:]
img_2 = combined_image[:,:, image_size_x-win_x:, 0:image_size_y+win_y]
img_3 = combined_image[:,:, image_size_x-win_x:, image_size_y-win_y:]


"""
Define model
"""
trained_network_directory = "/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/"
file_name_save = os.path.join(trained_network_directory, "2016-07-04_HeLa_set1_61x61_bn_0.h5")

model = sparse_bn_feature_net_61x61(batch_input_shape = (1,2,image_size_x+win_x, image_size_y+win_x), weights_path = file_name_save)

evaluate_model = K.function(
	[model.layers[0].input, K.learning_phase()],
	[model.layers[-1].output]
	) 

"""
Process image
"""

model_output[stack_iteration,:, 0:image_size_x-win_x, 0:image_size_y-win_y] = evaluate_model([img_0, 0])[0]
model_output[stack_iteration,:, 0:image_size_x-win_x, image_size_y-win_y:] = evaluate_model([img_1, 0])[0]
model_output[stack_iteration,:, image_size_x-win_x:, 0:image_size_y-win_y] = evaluate_model([img_2, 0])[0]
model_output[stack_iteration,:, image_size_x-win_x:, image_size_y-win_y:] = evaluate_model([img_3, 0])[0]


for j in xrange(1):
	cnnout_name = os.path.join(cnn_save_dir, 'keras_cyto_boundary_' + str(j) + r'.tif')
	tiff.imsave(cnnout_name,model_output[j,0,:,:])

	cnnout_name = os.path.join(cnn_save_dir, 'keras_cyto_interior_' + str(j) + r'.tif')
	tiff.imsave(cnnout_name,model_output[j,1,:,:])

	cnnout_name = os.path.join(cnn_save_dir, 'keras_cyto_background_' + str(j) + r'.tif')
	tiff.imsave(cnnout_name,model_output[j,2,:,:])



