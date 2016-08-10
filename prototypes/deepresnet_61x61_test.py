'''Train a simple deep CNN on a HeLa dataset.
GPU run command:
	THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python training_template.py

'''

from __future__ import print_function
from keras.optimizers import SGD, RMSprop, Adam

from cnn_functions import rate_scheduler, train_model_sample
from resnet_61x61 import deepresnet_61x61

#resnet_61x61
import os
import datetime
import numpy as np

batch_size = 256
n_classes = 3
n_epoch = 10

n_channels = 2
n_categories = 3

model = deepresnet_61x61(n_channels, n_categories)

dataset = "HeLa_all_61x61"
direc_save = "/home/nquach/DeepCell2/prototypes/trained_networks/"
direc_data = "/home/nquach/DeepCell2/training_data_npz/HeLa"
optimizer = Adam(lr = 0.001)
lr_sched = rate_scheduler(lr = 0.001, decay = 0.95)
expt = "deepresnet_61x61_test"

iterate = 1
train_model_sample(model = model, dataset = dataset, optimizer = optimizer, 
	expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
	direc_save = direc_save, 
	direc_data = direc_data, 
	lr_sched = lr_sched,
	rotate = True, flip = True, shear = 0)
