'''Train a simple deep CNN on a HeLa dataset.
GPU run command:
	THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python training_template.py

'''

from __future__ import print_function
from keras.optimizers import SGD, RMSprop

from cnn_functions import rate_scheduler, train_model_sample
from model_zoo import feature_net_61x61 as the_model

import os
import datetime
import numpy as np

batch_size = 256
n_classes = 3
n_epoch = 25

model = the_model(n_channels = 2, n_features = 3, reg = 1e-7, drop=0.5)
dataset = "HeLa_set1_set5_61x61"
direc_save = "/home/nquach/DeepCell2/trained_networks/"
direc_data = "/home/nquach/DeepCell2/training_data_npz/"
optimizer = RMSprop(lr = 0.001, rho = 0.95, epsilon = 1e-8)
lr_sched = rate_scheduler(lr = 0.001, decay = 0.95)
expt = "feature_net_61x61_drop_reg7"

iterate = 4
train_model_sample(model = model, dataset = dataset, optimizer = optimizer, 
	expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
	direc_save = direc_save, 
	direc_data = direc_data, 
	lr_sched = lr_sched,
	rotate = True, flip = True, shear = 0)