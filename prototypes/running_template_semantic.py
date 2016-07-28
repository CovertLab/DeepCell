'''Run a simple deep CNN on images.
GPU run command:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python running_template.py

'''

import h5py
import tifffile as tiff
from keras.backend.common import _UID_PREFIXES

from cnn_functions import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes, segment_nuclei, segment_cytoplasm, dice_jaccard_indices
from model_zoo import sparse_bn_feature_net_61x61 as cyto_fn
from model_zoo import sparse_bn_feature_net_61x61 as nuclear_fn


import os
import numpy as np


"""
Load data
"""
direc_name = '/home/vanvalen/DeepCell2/validation_data/MCF10A/'
data_location = os.path.join(direc_name, 'RawImages')

cyto_location = os.path.join(direc_name, 'Cytoplasm')
nuclear_location = os.path.join(direc_name, 'Nuclear')
mask_location = os.path.join(direc_name, 'Masks')

cyto_channel_names = ['Phase','DAPI']
nuclear_channel_names = ['DAPI']

trained_network_cyto_directory = "/home/vanvalen/DeepCell2/trained_networks/MCF10A"
trained_network_nuclear_directory = "/home/vanvalen/DeepCell2/trained_networks/Nuclear"

cyto_prefix = "2016-07-13_MCF10A_3T3_semantic_61x61_bn_feature_net_61x61_semantic_"
nuclear_prefix = "2016-07-12_nuclei_all_61x61_bn_"

win_cyto = 30
win_nuclear = 30

image_size_x, image_size_y = get_image_sizes(data_location, cyto_channel_names)
image_size_x /= 2
image_size_y /= 2

"""
Define model
"""

list_of_cyto_weights = []
for j in xrange(2):
	cyto_weights = os.path.join(trained_network_cyto_directory,  cyto_prefix + str(j) + ".h5")
	list_of_cyto_weights += [cyto_weights]

list_of_nuclear_weights = []
for j in xrange(4):
	nuclear_weights = os.path.join(trained_network_nuclear_directory,  nuclear_prefix + str(j) + ".h5")
	list_of_nuclear_weights += [nuclear_weights]

"""
Run model on directory
"""

nuclear_predictions = run_models_on_directory(data_location, nuclear_channel_names, nuclear_location, model_fn = nuclear_fn, 
	list_of_weights = list_of_nuclear_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
	win_x = win_nuclear, win_y = win_nuclear, std = False)

cytoplasm_predictions = run_models_on_directory(data_location, cyto_channel_names, cyto_location, model_fn = cyto_fn, 
	list_of_weights = list_of_cyto_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
	win_x = win_cyto, win_y = win_cyto, std = False)

"""
Refine segmentation with active contours
"""

nuclear_masks = segment_nuclei(nuclear_predictions, mask_location = mask_location, threshold = 0.85, area_threshold = 100, eccentricity_threshold = 0.95)
cytoplasm_masks = segment_cytoplasm(cytoplasm_predictions, nuclear_masks = nuclear_masks, mask_location = mask_location)


# """
# Compute validation metrics (optional)
# """
# direc_val = '/home/vanvalen/DeepCell2/validation_data/MCF10A/Validation/'
# imglist_val = nikon_getfiles(direc_val, 'validation_interior')

# val_name = os.path.join(direc_val, imglist_val[0]) 
# print val_name
# val = get_image(val_name)
# val = val[win_cyto:-win_cyto,win_cyto:-win_cyto]
# cyto = cytoplasm_masks[0,win_cyto:-win_cyto,win_cyto:-win_cyto]
# nuc = nuclear_masks[0,win_cyto:-win_cyto,win_cyto:-win_cyto]
# print val.shape, cyto.shape, nuc.shape


# dice_jaccard_indices(cyto, val, nuc)