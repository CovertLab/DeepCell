'''Train a simple deep CNN on a HeLa dataset.
GPU run command:
	THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python training_template.py

'''

from __future__ import print_function
from keras.optimizers import SGD, RMSprop

from cnn_functions import rate_scheduler, train_model_sample
from model_zoo import bn_feature_net_31x31 as the_model

import os
import datetime
import numpy as np

batch_size = 256
n_epoch = 25

dataset = "ecoli_all_31x31"
expt = "bn_feature_net_31x31"

direc_save = "/home/vanvalen/DeepCell2/trained_networks/"
direc_data = "/home/vanvalen/DeepCell2/training_data_npz/ecoli"

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr = 0.01, decay = 0.95)
class_weight = {0:1, 1:1, 2:1, 3:1}

for iterate in xrange(5):

	model = the_model(n_channels = 1, n_features = 3, reg = 1e-5)

	train_model_sample(model = model, dataset = dataset, optimizer = optimizer, 
		expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
		direc_save = direc_save, direc_data = direc_data, 
		lr_sched = lr_sched, class_weight = class_weight,
		rotate = True, flip = True, shear = False)

	del model
	from keras.backend.common import _UID_PREFIXES
	for key in _UID_PREFIXES.keys():
		_UID_PREFIXES[key] = 0

