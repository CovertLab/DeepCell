

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
#import matplotlib.pyplot as plt
import glob
import os
import fnmatch
import skimage as sk
import scipy as sp
from scipy import ndimage
from skimage import feature
from get_image import get_image
from skimage import morphology as morph

#import matplotlib.pyplot as plt
import coordinate_format as cf

# Define maximum number of training examples
max_training_examples = 10000000
window_size_x = 30
window_size_y = 30

# Load data

base_directory = '/home/nquach/DeepCell2/'
direc_name = os.path.join(base_directory, 'training_data', 'HeLa_joint')
file_name_save = os.path.join(base_directory, 'training_data_npz', 'HeLa_set1_set5_61x61.npz')
training_direcs = ['set1','set5']#,"cell2","cell3","cell4","cell5","cell6","cell7","cell8"] # , "set3/", "set4/", "set5/", "set6/"]
channel_names = ["nuclear","phase"]

# Specify the number of feature masks that are present
num_of_features = 2

# Some of the features that are edges we would like to do morphological
# dilation (disk of radius dil_radius) for data augmentation. This vector identifies 
# which features are edges and which ones arent (1 for an edge, 0 if not an edge)
is_edge_feature = [1,0]
dil_radius = 0

feature_names = []
img_format = ".png"
for j in xrange(num_of_features):
	feature_names += ["feature_" + str(j) + img_format]

num_direcs = len(training_direcs)
num_channels = len(channel_names)

imglist = []
for direc in training_direcs:
	imglist += os.listdir(os.path.join(direc_name, direc))

# Load one file to get image sizes
img_temp = get_image(os.path.join(direc_name,training_direcs[0],imglist[0]))
image_size_x, image_size_y = img_temp.shape

# Initialize arrays for the training images and the feature masks
channels = np.zeros((num_direcs, num_channels, image_size_x, image_size_y), dtype='float32')
feature_mask = np.zeros((num_direcs, num_of_features + 1, image_size_x, image_size_y))


# Load training images
direc_counter = 0
for direc in training_direcs:
	imglist = os.listdir(os.path.join(direc_name, direc))
	# print imglist
	channel_counter = 0
	# Load channels
	for channel in channel_names:
		for img in imglist: 
			if fnmatch.fnmatch(img, r'*' + channel + r'*'):
				channel_file = img
				channel_file = os.path.join(direc_name, direc, channel_file)
				channel_img = get_image(channel_file)
				
				p50 = np.percentile(channel_img, 50)
				channel_img /= p50
				avg_kernel = np.ones((2*window_size_x + 1, 2*window_size_y + 1))
				channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
				
				channels[direc_counter,channel_counter,:,:] = channel_img
				channel_counter += 1

	# Load feature mask
	for j in xrange(num_of_features):
		feature_name = "feature_" + str(j) + img_format
		print feature_name
		for img in imglist:
			if fnmatch.fnmatch(img, feature_name):
				feature_file = os.path.join(direc_name, direc,img)
				feature_img = get_image(feature_file)

				if np.sum(feature_img) > 0:
					feature_img /= np.amax(feature_img)

				if is_edge_feature[j] == 1:
					strel = sk.morphology.disk(dil_radius)
					feature_img = sk.morphology.binary_dilation(feature_img, selem = strel)

				feature_mask[direc_counter,j,:,:] = feature_img

	# Thin the augmented edges by subtracting the interior features.
	for j in xrange(num_of_features):
		if is_edge_feature[j] == 1:
			for k in xrange(num_of_features):
				if is_edge_feature[k] == 0:
					feature_mask[direc_counter,j,:,:] -= feature_mask[direc_counter,k,:,:]
			feature_mask[direc_counter,j,:,:] = feature_mask[direc_counter,j,:,:] > 0

	# Compute the mask for the background
	feature_mask_sum = np.sum(feature_mask[direc_counter,:,:,:], axis = 0)
	feature_mask[direc_counter,num_of_features,:,:] = 1 - feature_mask_sum

	direc_counter += 1


#Plot segementation results

'''fig,ax = plt.subplots(len(training_direcs),num_of_features+1, squeeze = False)
print ax.shape
for j in xrange(len(training_direcs)):
	ax[j,0].imshow(channels[j,0,:,:],cmap=plt.cm.gray,interpolation='nearest')
	def form_coord(x,y):
		return cf.format_coord(x,y,channels[j,0,:,:])
	ax[j,0].format_coord = form_coord
	ax[j,0].axes.get_xaxis().set_visible(False)
	ax[j,0].axes.get_yaxis().set_visible(False)

	for k in xrange(1,num_of_features+1):
		ax[j,k].imshow(feature_mask[j,k-1,:,:],cmap=plt.cm.gray,interpolation='nearest')
		def form_coord(x,y):
			return cf.format_coord(x,y,feature_mask[j,k-1,:,:])
		ax[j,k].format_coord = form_coord
		ax[j,k].axes.get_xaxis().set_visible(False)
		ax[j,k].axes.get_yaxis().set_visible(False)

plt.show()
'''

"""
Select points for training data
"""

# Find out how many example pixels exist for each feature and select the feature
# the fewest examples
feature_mask_trimmed = feature_mask[:,:,window_size_x+1:-window_size_x-1,window_size_y+1:-window_size_y-1] 
print feature_mask_trimmed.shape
feature_rows = []
feature_cols = []
feature_batch = []
feature_label = []

# We need to find the training data set with the least number of edge pixels. We will then sample
# that number of pixels from each of the training data sets (if possible)

edge_num = np.Inf
for j in xrange(feature_mask_trimmed.shape[0]):
	num_of_edge_pixels = 0
	for k in xrange(len(is_edge_feature)):
		if is_edge_feature[k] == 1:
			num_of_edge_pixels += np.sum(feature_mask_trimmed[j,k,:,:])

	if num_of_edge_pixels < edge_num:
		edge_num = num_of_edge_pixels

min_pixel_counter = edge_num

# feature_mask_summed = np.sum(feature_mask_trimmed, axis = 0)
# min_pixel_counter = np.sum(feature_mask_summed[0,:,:])

# for k in xrange(num_of_features + 1):
# 	min_pixel_counter = np.amin([min_pixel_counter, np.sum(feature_mask_summed[k,:,:])])

# Make sure we dont collect more than the maximum number of specified examples
# if (num_of_features+1)*min_pixel_counter > max_training_examples:
# 	min_pixel_counter = max_training_examples

print min_pixel_counter


for direc in xrange(channels.shape[0]):

	for k in xrange(num_of_features + 1):
		feature_counter = 0
		feature_rows_temp, feature_cols_temp = np.where(feature_mask[direc,k,:,:] == 1)

		# Check to make sure the features are actually present
		if len(feature_rows_temp) > 0:

			# Randomly permute index vector
			non_rand_ind = np.arange(len(feature_rows_temp))
			rand_ind = np.random.choice(non_rand_ind, size = len(feature_rows_temp), replace = False)

			for i in rand_ind:
				if feature_counter < min_pixel_counter:
					if (feature_rows_temp[i] - window_size_x > 0) and (feature_rows_temp[i] + window_size_x < image_size_x): 
						if (feature_cols_temp[i] - window_size_y > 0) and (feature_cols_temp[i] + window_size_y < image_size_y):
							feature_rows += [feature_rows_temp[i]]
							feature_cols += [feature_cols_temp[i]]
							feature_batch += [direc]
							feature_label += [k]
							feature_counter += 1

feature_rows = np.array(feature_rows,dtype = 'int32')
feature_cols = np.array(feature_cols,dtype = 'int32')
feature_batch = np.array(feature_batch, dtype = 'int32')
feature_label = np.array(feature_label, dtype = 'int32')


print feature_rows.shape, feature_cols.shape, feature_batch.shape, feature_label.shape
print np.amax(feature_label)

print np.bincount(feature_batch)

# Randomly select training points 
if len(feature_rows) > max_training_examples:
	non_rand_ind = np.arange(len(feature_rows))
	rand_ind = np.random.choice(non_rand_ind, size = max_training_examples, replace = False)

	feature_rows = feature_rows[rand_ind]
	feature_cols = feature_cols[rand_ind]
	feature_batch = feature_batch[rand_ind]
	feature_label = feature_label[rand_ind]

# Randomize
non_rand_ind = np.arange(len(feature_rows))
rand_ind = np.random.choice(non_rand_ind, size = len(feature_rows), replace = False)

feature_rows = feature_rows[rand_ind]
feature_cols = feature_cols[rand_ind]
feature_batch = feature_batch[rand_ind]
feature_label = feature_label[rand_ind]

print np.bincount(feature_batch)

np.savez(file_name_save, channels = channels, y = feature_label, batch = feature_batch, pixels_x = feature_rows, pixels_y = feature_cols, win_x = window_size_x, win_y = window_size_y)

