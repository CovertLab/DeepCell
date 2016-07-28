import os
import glob
import numpy as np
import tifffile as tiff
from numpy.fft import fft2, ifft2, fftshift
from nikon_getfiles import nikon_getfiles
from skimage.io import imread

def get_image(file_name):
	if '.tif' in file_name:
		im = np.float32(tiff.TIFFfile(file_name).asarray())
	else:
		im = np.float32(imread(file_name))
	return im