'''Train a simple deep CNN on a HeLa dataset.
GPU run command:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

'''

#from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Input, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
import keras.backend as K

from cnn_functions import get_data_siamese, contrastive_loss, morgan_loss, same_loss, ImageDataGenerator, randomly_rotate_array
from model_zoo import siamese_net_51x51_test, simple_siamese
from get_data_siamese_movie import get_data_siamese_movie

import os
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)
'''
reg = 1e-5
drop = 0
init = 'he_normal'
right_twin = Sequential()
right_twin.add(Convolution2D(32, 4, 4, name = 'layer1', init = init, border_mode='valid', input_shape=(1, 51, 51), W_regularizer = l2(reg)))
right_twin.add(Activation('relu'))
right_twin.add(MaxPooling2D(pool_size=(2, 2)))
right_twin.add(Convolution2D(64, 3, 3, name = 'layer2', init = init, border_mode='valid', W_regularizer = l2(reg)))
right_twin.add(Activation('relu'))
right_twin.add(Convolution2D(64, 3, 3, name = 'layer3', init = init, border_mode='valid', W_regularizer = l2(reg)))
right_twin.add(Activation('relu'))
right_twin.add(MaxPooling2D(pool_size=(2, 2)))
right_twin.add(Convolution2D(64, 3, 3, name = 'layer4', init = init, border_mode='valid', W_regularizer = l2(reg)))
right_twin.add(Activation('relu'))
right_twin.add(MaxPooling2D(pool_size=(2, 2)))
right_twin.add(Flatten())
right_twin.add(Dense(256, name = 'layer5', init = init, W_regularizer = l2(reg)))
if drop > 0:
	right_twin.add(Dropout(drop))
right_twin.add(Activation('relu'))
right_twin.add(Dense(256, name = 'layer6', init = init, W_regularizer = l2(reg)))
if drop > 0:
	right_twin.add(Dropout(drop))
right_twin.add(Activation('relu'))

left_twin = Sequential()
left_twin.add(Convolution2D(32, 4, 4, name = 'layer1', init = init, border_mode='valid', input_shape=(1, 51, 51), W_regularizer = l2(reg)))
left_twin.add(Activation('relu'))
left_twin.add(MaxPooling2D(pool_size=(2, 2)))
left_twin.add(Convolution2D(64, 3, 3, name = 'layer2', init = init, border_mode='valid', W_regularizer = l2(reg)))
left_twin.add(Activation('tanh'))
left_twin.add(Convolution2D(64, 3, 3, name = 'layer3', init = init, border_mode='valid', W_regularizer = l2(reg)))
left_twin.add(Activation('tanh'))
left_twin.add(MaxPooling2D(pool_size=(2, 2)))
left_twin.add(Convolution2D(64, 3, 3, name = 'layer4', init = init, border_mode='valid', W_regularizer = l2(reg)))
left_twin.add(Activation('tanh'))
left_twin.add(MaxPooling2D(pool_size=(2, 2)))
left_twin.add(Flatten())
left_twin.add(Dense(256, name = 'layer5', init = init, W_regularizer = l2(reg)))
if drop > 0:
	left_twin.add(Dropout(drop))
left_twin.add(Activation('tanh'))
left_twin.add(Dense(256, name = 'layer6', init = init, W_regularizer = l2(reg)))
if drop > 0:
	left_twin.add(Dropout(drop))
right_twin.add(Activation('tanh'))


#merged = Merge([right_twin, right_twin], mode = euclidean_distance, output_shape=eucl_dist_output_shape)

model = Sequential()
model.add(Merge([right_twin, right_twin], mode = euclidean_distance, output_shape=eucl_dist_output_shape))
'''
model = siamese_net_51x51_test()

print "Loaded model"

batch_size = 256
nb_epoch = 10

trained_network_directory = "/home/nquach/siamese/trained_networks/"
data_directory = "/home/nquach/siamese/training_data_npz/"
training_data_file_name = os.path.join(data_directory, "nuclei_movie_siamese_51x51.npz")

file_name_save = os.path.join(trained_network_directory, "2016-07-18_nuclei_movie_siamese_51x51_0.h5")
loss_name_save = os.path.join(trained_network_directory, "2016-07-18_nuclei_movie_siamese_51x51_0.npz")
train_dict, test_dict = get_data_siamese_movie(training_data_file_name)

# input image dimensions
img_rows, img_cols = 51, 51
img_channels = 1 

print "Loaded data"
# Train the model
# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr = 0.001, rho = 0.95, epsilon = 1e-8)

model.compile(loss=contrastive_loss,optimizer=rmsprop)

print np.unique(train_dict['input1'][0])
print "Compiled model"

loss_hist = model.fit([train_dict['input1'], train_dict['input2']], train_dict['labels'], nb_epoch = nb_epoch, batch_size = batch_size)
np.savez(loss_name_save, loss_history = loss_hist.history)

print "Trained model"
#model.save_weights(file_name_save)

distances = model.predict([test_dict['input1'], test_dict['input2']], batch_size = batch_size)

same = []
diff = []
print "Predicted labels"
print distances

for distance, label in zip(distances, test_dict['labels']):
#for distance, label in zip(distances, labels)
	if label == 0:
		diff += [distance]
	else:
		same += [distance]


diff_list = np.array(diff)
same_list = np.array(same)
sns.set_style("white")
sns.set_context("poster")
sns.distplot(diff_list, label = 'Different nuclei', color = "red")
sns.distplot(same_list, label = 'Same nuclei', color = "green")
plt.xlabel("Euclidian distance")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('/home/nquach/DeepCell2/prototypes/plots/siamese_test.pdf')

