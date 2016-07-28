'''Run a simple deep CNN on images.
GPU run command:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_feature_net_61x61.py
'''

import h5py
import tifffile as tiff

from cnn_functions import run_model_on_directory, get_image_sizes
from model_zoo import sparse_bn_feature_net_61x61 as the_model

import os
import datetime
import numpy as np
import theano

"""
Load data
"""
direc_name = '/home/vanvalen/DeepCell2/testing_data/HeLa/set1/'
data_location = os.path.join(direc_name, 'RawImages')
output_location = os.path.join(direc_name, 'Output')
channel_names = ['Phase','Far-red']

win_x = 30
win_y = 30

image_size_x, image_size_y = get_image_sizes(data_location, channel_names)
image_size_x /= 2
image_size_y /= 2

"""
Define model
"""
trained_network_directory = "/home/vanvalen/DeepCell2/trained_networks/"
file_name_save = os.path.join(trained_network_directory, "2016-07-12_HeLa_all_61x61_bn_shear_0.h5")

model = the_model(batch_input_shape = (1,2,image_size_x+win_x, image_size_y+win_x), weights_path = file_name_save)

"""
Run model on directory
"""

run_model_on_directory(data_location, channel_names, output_location, model = model, win_x = win_x, win_y = win_y, std = False)