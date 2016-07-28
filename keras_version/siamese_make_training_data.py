"""
Created on 05/06/2015

@author: vanvalen

Create training data - this file imports the masks that were manually segmented
and crops out a small region around each pixel identified as an edge.
It also picks an equal number of interior and exterior pixels and crops out a corresponding
region around them as well. The images are saved as tiffs in a directory
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
from get_image import get_image
from skimage import morphology as morph
from skimage.measure import regionprops, label

import matplotlib.pyplot as plt
import coordinate_format as cf

# Define maximum number of training examples
max_training_examples = 10000000
window_size_x = 25
window_size_y = 25

# Load data

direc_name = r'/home/vanvalen/ImageAnalysis/DeepCell2/training_data/nuclei/'
direc_save = "/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/"
file_name_save = os.path.join(direc_save, r'nuclei_all_siamese_51x51.npz')
training_direcs = ['cell1/', 'cell2/','cell3/', 'cell4/', 'cell5/', 'cell6/', 'cell7/', 'cell8/']
channel_names = ['nuclear']
edge_name = 'feature_0'
int_name = 'feature_1'
tiff_end = '.tif'
png_end = '.png'
num_direcs = len(training_direcs)
num_channels = len(channel_names)

imglist = []
for direc in training_direcs:
	imglist += os.listdir(direc_name + direc)

# Load one file to get image sizes
phase_temp = get_image(direc_name + training_direcs[0] + imglist[0])

image_size_x, image_size_y = phase_temp.shape
channels = np.zeros((num_direcs,num_channels,image_size_x,image_size_y), dtype='float32')
interior_mask = np.zeros((num_direcs,image_size_x,image_size_y))
exterior_mask = np.zeros((num_direcs,image_size_x,image_size_y))
edge_mask = np.zeros((num_direcs,image_size_x,image_size_y))

# Load phase images

direc_counter = 0
for direc in training_direcs:
	print direc
	imglist = os.listdir(direc_name + direc)
	print imglist
	channel_counter = 0
	# Load channels
	for channel in channel_names:
		print channel
		for img in imglist: 

			if fnmatch.fnmatch(img, r'*' + channel + r'*' + tiff_end) or fnmatch.fnmatch(img, r'*' +channel + r'*' + png_end):
				print img

				channel_file = img
				channel_file = direc_name + direc + channel_file
				channel_img = get_image(channel_file).astype('float32')
				
				# p50 = np.percentile(channel_img, 50)
				# channel_img /= p50

				# avg_kernel = np.ones((2*window_size_x+1,2*window_size_y+1))
				# channel_mean = ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
				# channel_img -= channel_mean

				channels[direc_counter,channel_counter,:,:] = channel_img

				channel_counter += 1

	# Load interior mask
	for img in imglist:
		if fnmatch.fnmatch(img,int_name + r'*' + png_end):
			int_file = img
			int_file = direc_name + direc + int_file
			int_img = get_image(int_file)
			int_img = int_img/np.amax(int_img)
			interior_mask[direc_counter,:,:] = int_img

	# Load edge mask
	for img in imglist:
		if fnmatch.fnmatch(img,edge_name + r'*' + png_end):
			edge_file = img
			edge_file = direc_name + direc + edge_file
			edge_img = get_image(edge_file)
			edge_img = edge_img/np.amax(edge_img)
			edge_mask[direc_counter,:,:] = edge_img
	
	edge_mask[direc_counter,:,:] -= interior_mask[direc_counter,:,:]
	edge_mask[direc_counter,:,:] = edge_mask[direc_counter,:,:] > 0

	exterior_mask[direc_counter,:,:] = 1 - edge_mask[direc_counter,:,:] - interior_mask[direc_counter,:,:]

	direc_counter += 1

	total_mask = edge_mask + 2*interior_mask

	# Label interior mask
	int_mask_label = label(interior_mask)

	#Plot segementation results
	# fig,ax = plt.subplots(1,4)

	# ax[0].imshow(channels[0,0,:,:],cmap=plt.cm.gray,interpolation='nearest')
	# def form_coord(x,y):
	# 	return cf.format_coord(x,y,channels[0,0,:,:])
	# ax[0].format_coord = form_coord
	# ax[0].axes.get_xaxis().set_visible(False)
	# ax[0].axes.get_yaxis().set_visible(False)

	# ax[1].imshow(edge_mask[0,:,:],cmap=plt.cm.gray,interpolation='nearest')
	# def form_coord(x,y):
	# 	return cf.format_coord(x,y,edge_mask[0,:,:])
	# ax[1].format_coord = form_coord
	# ax[1].axes.get_xaxis().set_visible(False)
	# ax[1].axes.get_yaxis().set_visible(False)

	# ax[2].imshow(interior_mask[0,:,:],cmap=plt.cm.gray,interpolation='nearest')
	# def form_coord(x,y):
	# 	return cf.format_coord(x,y,interior_mask[0,:,:])
	# ax[2].format_coord = form_coord
	# ax[2].axes.get_xaxis().set_visible(False)
	# ax[2].axes.get_yaxis().set_visible(False)

	# ax[3].imshow(exterior_mask[0,:,:],cmap=plt.cm.gray,interpolation='nearest')
	# def form_coord(x,y):
	# 	return cf.format_coord(x,y,total_mask[0,:,:])
	# ax[3].format_coord = form_coord
	# ax[3].axes.get_xaxis().set_visible(False)
	# ax[3].axes.get_yaxis().set_visible(False)

	# plt.show()

# Properly label all of the nuclear masks
label_mask = np.zeros(interior_mask.shape, dtype = np.int)
running_max = 0
for j in xrange(len(training_direcs)):
	nuc_label = label(interior_mask[j,:,:], connectivity = 1) + running_max
	nuc_label[nuc_label == running_max] = 0
	label_mask[j,:,:] = nuc_label
	running_max = np.amax(nuc_label)
	print running_max

# Identify centroids of each nucleus
image_list = []
id_list = []

counter = 0  
for j in xrange(len(training_direcs)):
	nuc_label = label_mask[j,:,:]
	props = regionprops(nuc_label)
	for prop in props:
		cent_x = np.int(np.round(prop.centroid[0]))
		cent_y = np.int(np.round(prop.centroid[1]))
		if prop.area > 50 and cent_x-window_size_x > 0 and cent_x+window_size_x < interior_mask.shape[1] and cent_y-window_size_y > 0 and cent_y+window_size_y < interior_mask.shape[2]:
			nuc_mask = nuc_label[cent_x-window_size_x:cent_x+window_size_x+1, cent_y-window_size_y:cent_y+window_size_y+1] == prop.label
			cropped_img = channels[j,:,cent_x-window_size_x:cent_x+window_size_x+1, cent_y-window_size_y:cent_y+window_size_y+1]

			cropped_img -= np.mean(cropped_img)
			cropped_img /= np.std(cropped_img)

			img = cropped_img*nuc_mask

			image_list += [img]
			id_list += [counter]
			counter += 1

			# print j, np.max(img), np.min(img), np.mean(img), np.std(img)

			# fig,ax = plt.subplots(1,2, squeeze = False)

			# ax[0,0].imshow(img[0,:,:],cmap=plt.cm.gray,interpolation='nearest')
			# def form_coord(x,y):
			# 	return cf.format_coord(x,y,img[0,:,:])
			# ax[0,0].format_coord = form_coord
			# ax[0,0].axes.get_xaxis().set_visible(False)
			# ax[0,0].axes.get_yaxis().set_visible(False)

			# ax[0,1].imshow(nuc_mask,cmap=plt.cm.gray,interpolation='nearest')
			# def form_coord(x,y):
			# 	return cf.format_coord(x,y,nuc_mask)
			# ax[0,1].format_coord = form_coord
			# ax[0,1].axes.get_xaxis().set_visible(False)
			# ax[0,1].axes.get_yaxis().set_visible(False)
			# plt.show()

image_list = np.stack(image_list, axis = 0)
np.savez(file_name_save, image_list = image_list, id_list = id_list)

