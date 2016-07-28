import numpy as np
import random

def get_data_siamese_movie(file_name):
	training_data = np.load(file_name)
	input1 = training_data['input1']
	input2 = training_data['input2']
	labels = training_data['labels']

	print len(input1), len(input2), len(labels)
	
	total_batch_size = len(labels)
	num_test = np.int32(np.floor(total_batch_size/200))
	num_train = np.int32(total_batch_size - num_test)
	full_batch_size = np.int32(num_test + num_train)

	print "total batch size:", total_batch_size
	print "num_test:", num_test
	print "num_train:" , num_train
	print "full_batch_size", full_batch_size

	#input1_train = np.stack(input1[0:num_train], axis = 0)
	#input2_train = np.stack(input2[0:num_train], axis = 0)
	input1_train = input1[0:num_train]
	input2_train = input2[0:num_train]
	labels_train = labels[0:num_train]

	print input2_train.shape
	print labels_train.shape

	input1_test = input1[num_train + 1 : num_train + num_test]
	input2_test = input2[num_train + 1 : num_train + num_test]
	labels_test = labels[num_train + 1 : num_train + num_test]

	print input2_test.shape
	print labels_test.shape

	train_set = {"input1" : input1_train, "input2" : input2_train, "labels" : labels_train}
	test_set = {"input1" : input1_test, "input2" : input2_test, "labels" : labels_test}
	return train_set, test_set