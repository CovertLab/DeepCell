'''Train a simple deep CNN on a HeLa dataset.
GPU run command:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
import keras.backend as K

from cnn_functions import get_data_siamese, contrastive_loss, morgan_loss, same_loss, ImageDataGenerator, randomly_rotate_array
from model_zoo import siamese_net_51x51, simple_siamese

import os
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

batch_size = 256
nb_epoch = 2

trained_network_directory = "/home/vanvalen/DeepCell2/trained_networks/"
data_directory = "/home/vanvalen/DeepCell2/training_data_npz/"
training_data_file_name = os.path.join(data_directory, "nuclei_all_siamese_51x51.npz")

file_name_save = os.path.join(trained_network_directory, "2016-07-07_nuclei_all_siamese_51x51_0.h5")
train_dict, (test_input_dict, test_label_dict) = get_data_siamese(training_data_file_name)

# input image dimensions
img_rows, img_cols = 51, 51
img_channels = 1

# the data, shuffled and split between train and test sets
print('X_train shape:', train_dict["image_list"].shape)
print(len(train_dict["ids"]), 'train samples')
print(len(test_label_dict["lambda_1"]), 'test samples')

model = siamese_net_51x51()

# Train the model
# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr = 0.001, rho = 0.95, epsilon = 1e-8)

model.load_weights(file_name_save)
model.compile(loss=contrastive_loss,
			  optimizer=rmsprop,
			  )

evaluate_model = K.function(
	[model.layers[0].input, model.layers[1].input, K.learning_phase()],
	[model.layers[-1].output]
	) 

input_1 = test_input_dict['input_1']
input_2 = test_input_dict['input_2']
same = test_label_dict['lambda_1']

dist_list = []

for j in xrange(len(input_1)):
	in_1 = randomly_rotate_array(np.expand_dims(input_1[j], axis = 0))
	in_2 = randomly_rotate_array(np.expand_dims(input_2[j], axis = 0))
	dist = evaluate_model([in_1, in_2, 0])[0][0][0]
	# print(dist, same[j])
	dist_list += [dist]

same_list = []
diff_list = []

for d, s in zip(dist_list, same):
	if s == 0:
		diff_list += [d]
	elif s == 1:
		same_list += [d]

diff_list = np.array(diff_list)
same_list = np.array(same_list)
sns.set_style("white")
sns.set_context("poster")
sns.distplot(diff_list, label = 'Different nuclei', color = "red")
sns.distplot(same_list, label = 'Same nuclei', color = "green")
plt.xlabel("Euclidian distance")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('plots/siamese_test.pdf')
plt.show()


