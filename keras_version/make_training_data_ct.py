"""
Created on 06/23/2016

@author: vanvalen

Create training data - this file imports the masks that were manually segmented
and crops out a small region around each pixel identified as an edge.
It also picks an equal number of interior and exterior pixels and crops out a corresponding
region around them as well. 
"""

"""
Import packages
"""


import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import fnmatch
import skimage as sk
import scipy as sp
import pandas as pd
from scipy import ndimage
from skimage import feature
from cnn_functions import get_image
from cnn_functions import format_coord as cf
from skimage import morphology as morph
import matplotlib.pyplot as plt
from skimage.transform import resize


# Define maximum number of training examples
max_training_examples = 10000000
window_size_x = 174
window_size_y = 174

# Load data
direc_name = '/home/vanvalen/DeepCell/training_data/CT-chest/'
file_name_save = os.path.join('/home/vanvalen/DeepCell/training_data_npz/CT-chest/', 'CT-chest_174x174.npz')
training_direcs = ["set1/", "set2/", "set3/", "set4/", "set5/", "set6/", "set7/", "set8/"]
channel_names = ["CT"]

num_direcs = len(training_direcs)
num_channels = len(channel_names)

imglist = []
img_name_list = []
for direc in training_direcs:
	imglist += os.listdir(os.path.join(direc_name, direc))

# Set image sizes
image_size_x, image_size_y = (174,174)

# Initialize arrays for the training images and the feature masks
channels = np.zeros((num_direcs, num_channels, image_size_x, image_size_y), dtype='float32')

# Load training images
direc_counter = 0
for direc in training_direcs:
	imglist = os.listdir(os.path.join(direc_name, direc))
	channel_counter = 0

	# Load channels
	for channel in channel_names:
		for img in imglist: 
			if fnmatch.fnmatch(img, r'*' + channel + r'*.tif'):
				channel_file = img
				img_name_list += [os.path.basename(img)]
				channel_file = os.path.join(direc_name, direc, channel_file)
				channel_img = get_image(channel_file)
		
				channels[direc_counter,channel_counter,:,:] = channel_img[0:174,0:174]
				channel_counter += 1

	direc_counter += 1

# Identify which images have cancer or not
metadata_file = os.path.join(direc_name, "OnughaDatabase-TEST17.xlsx")
cancer_list = ["Adenocarcinoma", "Squamous Cell Carcinoma"]
xl = pd.ExcelFile(metadata_file)
dataframe = xl.parse('Sheet1')
indexed_dataframe = dataframe.set_index('Pt ID')
y = []
for file_name in img_name_list:
	histology = indexed_dataframe[file_name, "Histologic Type"]
	if histology in cancer_list:
		y += [1]
	else:
		y += [0]


np.savez(file_name_save, channels = channels, y = y)

