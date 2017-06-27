'''Run a simple deep CNN on images.
GPU run command:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python running_template.py

'''

import h5py
import tifffile as tiff
from keras.backend.common import _UID_PREFIXES

from cnn_functions import nikon_getfiles, get_image, run_models_on_lsm, get_image_sizes, segment_nuclei, segment_cytoplasm, dice_jaccard_indices
from model_zoo import sparse_bn_feature_net_31x31 as cyto_fn

import os
import numpy as np
from pylsm import lsmreader

"""
Load data
"""
direc_name = '/home/vanvalen/Data/mouse_embryos/'
data_location = os.path.join(direc_name, 'RawData')
mask_location = os.path.join(direc_name, 'Masks')
save_location = os.path.join(data_location, 'cnn_output')
# file_name = '4Jun14_E3_25_8.lsm'
file_name = '6Feb14FGFonCD1_1_Control_2.lsm'
# file_name = '1May14FGFonCD1_9_FGF4_1.lsm'

lsm_file = os.path.join(data_location, file_name)

trained_network_cyto_directory = "/home/vanvalen/DeepCell/trained_networks/mouse_embryos"
cyto_prefix = "2017-06-27_mouse_embryos_31x31_bn_feature_net_31x31_"
win_cyto = 15

raw_image_file = lsmreader.Lsmimage(lsm_file)
raw_image_file.open()

image_size_x = raw_image_file.image['data'][0].shape[0]
image_size_y = raw_image_file.image['data'][0].shape[1]
image_size_z = raw_image_file.image['data'][0].shape[2]
num_channels = len(raw_image_file.image['data'])


"""
Define model
"""

list_of_cyto_weights = []
for j in xrange(1):
	cyto_weights = os.path.join(trained_network_cyto_directory,  cyto_prefix + str(j) + ".h5")
	list_of_cyto_weights += [cyto_weights]

print list_of_cyto_weights

"""
Run model on lsm file
"""

cytoplasm_predictions = run_models_on_lsm(lsm_file, save_location, n_features = 3, model_fn = cyto_fn, 
	list_of_weights = list_of_cyto_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
	win_x = win_cyto, win_y = win_cyto, std = False, split = False)


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