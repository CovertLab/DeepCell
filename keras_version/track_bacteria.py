import h5py
import tifffile as tiff
from keras.backend.common import _UID_PREFIXES

from cnn_functions import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes, segment_nuclei, segment_cytoplasm, dice_jaccard_indices
from model_zoo import sparse_bn_feature_net_31x31 as cyto_fn

import os
import numpy as np

from cnn_functions import get_image, align_images, crop_images, make_tracks, get_lineage
from cnn_functions import create_masks, plot_lineage, plot_lineage_numbers, plot_lineage_total
import matplotlib as mpl
import scipy
mpl.rcParams['pdf.fonttype'] = 42

direc_name = "/home/vanvalen/Data/ecoli"
image_dir = os.path.join(direc_name, "RawImages/")
align_dir = os.path.join(direc_name, "Align/")
cnn_save_dir = os.path.join(direc_name, "Cytoplasm/")
mask_dir = os.path.join(direc_name, "Masks/")
region_dir = os.path.join(direc_name, "Regions/")
cropped_dir = os.path.join(direc_name, "Cropped/")
track_dir = os.path.join(direc_name, "Tracks/")
cost_dir = os.path.join(direc_name, "Cost_Matrices/")

# Load Regions
region_file = np.load(os.path.join(region_dir, 'regions_save.npz'))
regions_save = region_file['regions_save']

total_no_of_frames = 32

# Load phase images
list_of_tracks = []
for chunk in [5]: #xrange(1,len(regions_save)):
	print chunk
	tracks = make_tracks(regions = regions_save[chunk], direc_save = track_dir, start_frame = 1, end_frame = 31, direc_cost_save = cost_dir)
	list_of_tracks += [tracks]

file_name_save = 'list_of_tracks'
np.savez(os.path.join(track_dir, file_name_save), tracks = list_of_tracks)