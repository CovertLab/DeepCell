'''Run a simple deep CNN on images.
GPU run command:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python running_template.py

'''

import h5py
import tifffile as tiff
from keras.backend.common import _UID_PREFIXES

from cnn_functions import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes, segment_nuclei, segment_cytoplasm, dice_jaccard_indices
from model_zoo import sparse_bn_feature_net_31x31 as cyto_fn
from model_zoo import sparse_bn_feature_net_61x61 as nuclear_fn


import os
import numpy as np


"""
Load data
"""
direc_name = '/home/vanvalen/Data/ecoli'
data_location = os.path.join(direc_name, 'Align')
cyto_location = os.path.join(direc_name, 'Cytoplasm')
nuclear_location = os.path.join(direc_name, 'Nuclear')
mask_location = os.path.join(direc_name, 'Masks')

cyto_channel_names = ['Pos3']
nuclear_channel_names = ['DAPI']

trained_network_cyto_directory = "/home/vanvalen/DeepCell2/trained_networks/"
trained_network_nuclear_directory = "/home/vanvalen/DeepCell2/trained_networks/Nuclear"

cyto_prefix = "2016-08-02_ecoli_all_31x31_bn_feature_net_31x31_"
nuclear_prefix = "2016-07-12_nuclei_all_61x61_bn_feature_net_61x61_"

win_cyto = 15
win_nuclear = 30

image_size_x, image_size_y = get_image_sizes(data_location, cyto_channel_names)

"""
Define model
"""

list_of_cyto_weights = []
for j in xrange(5):
	cyto_weights = os.path.join(trained_network_cyto_directory,  cyto_prefix + str(j) + ".h5")
	list_of_cyto_weights += [cyto_weights]

list_of_nuclear_weights = []
for j in xrange(5):
	nuclear_weights = os.path.join(trained_network_nuclear_directory,  nuclear_prefix + str(j) + ".h5")
	list_of_nuclear_weights += [nuclear_weights]

"""
Run model on directory
"""

cytoplasm_predictions = run_models_on_directory(data_location, cyto_channel_names, cyto_location, n_features = 3, model_fn = cyto_fn, 
	list_of_weights = list_of_cyto_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
	win_x = win_cyto, win_y = win_cyto, std = False, split = False)

# nuclear_predictions = run_models_on_directory(data_location, nuclear_channel_names, nuclear_location, model_fn = nuclear_fn, 
# 	list_of_weights = list_of_nuclear_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
# 	win_x = win_nuclear, win_y = win_nuclear, std = False, split = False)

"""
Refine segmentation with active contours
"""

# nuclear_masks = segment_nuclei(img = None, color_image = True, load_from_direc = nuclear_location, mask_location = mask_location, area_threshold = 100, solidity_threshold = 0, eccentricity_threshold = 1)
# cytoplasm_masks = segment_cytoplasm(img = None, load_from_direc = cyto_location, color_image = True, nuclear_masks = nuclear_masks, mask_location = mask_location, smoothing = 1, num_iters = 120)


"""
Compute validation metrics (optional)
"""
# direc_val = os.path.join(direc_name, 'Validation')
# imglist_val = nikon_getfiles(direc_val, 'feature_1')

# val_name = os.path.join(direc_val, imglist_val[0]) 
# print val_name
# val = get_image(val_name)
# val = val[win_cyto:-win_cyto,win_cyto:-win_cyto]
# cyto = cytoplasm_masks[0,win_cyto:-win_cyto,win_cyto:-win_cyto]
# nuc = nuclear_masks[0,win_cyto:-win_cyto,win_cyto:-win_cyto]
# print val.shape, cyto.shape, nuc.shape


# dice_jaccard_indices(cyto, val, nuc)