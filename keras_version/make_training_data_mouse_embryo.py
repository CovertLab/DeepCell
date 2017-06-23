"""
Created on 06/23/2017

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
from scipy import ndimage
from skimage import feature
from cnn_functions import get_image
from cnn_functions import format_coord as cf
from skimage import morphology as morph
from skimage.filters import sobel, roberts
import matplotlib.pyplot as plt
from skimage.transform import resize
from pylsm import lsmreader


# Define maximum number of training examples
max_training_examples = 10000000
window_size_x = 30
window_size_y = 30

# Load data
direc_name = '/home/vanvalen/Data/mouse_embryos/'
file_name_save = os.path.join('/home/vanvalen/DeepCell/training_data_npz/mouse_embryos/', 'mouse_embryos_61x61.npz')
raw_direc = os.path.join(direc_name, "RawData")
seg_direc = os.path.join(direc_name, "SegmentationMasks")

list_of_raw_files = os.listdir(raw_direc)
list_of_seg_files = os.listdir(seg_direc)

channels_list = []
features_list = []
for raw_name in [list_of_raw_files[0]]:
	# try:
	file_base = os.path.splitext(os.path.basename(raw_name))[0]
	seg_name = file_base + "_channel=0001_frame=0001_segmentation.tiff"

	raw_file_name = os.path.join(raw_direc, raw_name)
	seg_file_name = os.path.join(seg_direc, seg_name)
	raw_image_file = lsmreader.Lsmimage(raw_file_name)

	# try:
	raw_image_file.open()
	print raw_file_name
	vx = raw_image_file.image['data'][0].shape[0]
	vy = raw_image_file.image['data'][0].shape[1]
	vz = raw_image_file.image['data'][0].shape[2]
	num_channels = len(raw_image_file.image['data'])

	channels_temp = np.zeros((vz, num_channels, vx, vy), dtype = 'float32')
	for zpos in xrange(vz):
		for channel in xrange(num_channels):
			channel_img = np.flipud(raw_image_file.get_image(stack = zpos, channel = channel))
			channel_img = np.float32(channel_img)
			p50 = np.percentile(channel_img, 50)
			channel_img /= p50

			avg_kernel = np.ones((2*window_size_x + 1, 2*window_size_y + 1))
			channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size

			channels_temp[zpos, channel, :, :] = channel_img

	seg_image = get_image(seg_file_name)

	seg_edge = np.zeros(seg_image.shape)
	seg_interior = np.zeros(seg_image.shape)
	seg_background = np.zeros(seg_image.shape)
	features_temp = np.zeros((vz, 3, vx, vy))

	for zpos in xrange(vz):
		seg_img = seg_image[zpos,:,:]
		seg_img_erode = morph.erosion(seg_img, morph.disk(2))

		seg_edge_temp = seg_img-seg_img_erode
		seg_edge_temp[seg_edge_temp>0] = 1
		seg_edge[zpos,:,:] = seg_edge_temp

		seg_int_temp = seg_img
		seg_int_temp[seg_int_temp > 1] = 1
		seg_int_temp -= seg_edge_temp
		seg_int_temp[seg_int_temp>0] = 1
		seg_int_temp[seg_int_temp<0] = 0
		seg_interior[zpos,:,:] = seg_int_temp

		seg_back_temp = np.ones(seg_int_temp.shape) - seg_int_temp - seg_edge_temp
		seg_back_temp[seg_back_temp<0] = 0
		seg_background[zpos,:,:] = seg_back_temp

		features_temp[zpos, 0, :, :]= seg_edge_temp
		features_temp[zpos, 1, :, :]= seg_int_temp
		features_temp[zpos, 2, :, :]= seg_background_temp

	channels_list += [channels_temp]
	features_list += [features_temp]

	# except:
	# 	pass

print channels_list
channels = np.concatenate(channels_list)
feature_mask = np.concatenate(features_list)


# #Plot segementation results

# fig,ax = plt.subplots(len(training_direcs),num_of_features+2, squeeze = False)
# print ax.shape
# for j in xrange(len(training_direcs)):
# 	ax[j,0].imshow(channels[j,0,:,:],cmap=plt.cm.gray,interpolation='nearest')
# 	def form_coord(x,y):
# 		return cf(x,y,channels[j,0,:,:])
# 	ax[j,0].format_coord = form_coord
# 	ax[j,0].axes.get_xaxis().set_visible(False)
# 	ax[j,0].axes.get_yaxis().set_visible(False)

# 	for k in xrange(1,num_of_features+2):
# 		ax[j,k].imshow(feature_mask[j,k-1,:,:],cmap=plt.cm.gray,interpolation='nearest')
# 		def form_coord(x,y):
# 			return cf(x,y,feature_mask[j,k-1,:,:])
# 		ax[j,k].format_coord = form_coord
# 		ax[j,k].axes.get_xaxis().set_visible(False)
# 		ax[j,k].axes.get_yaxis().set_visible(False)

# plt.show()


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
print np.sum(feature_label == 0), np.sum(feature_label == 1), np.sum(feature_label == 2), np.sum(feature_label == 3)
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