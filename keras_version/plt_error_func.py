
import numpy as np 
import matplotlib.pyplot as plt 

#loss_hist_file is full path to .npz loss history file
#saved_direc is full path to directory where you want to save the plot
#plot_name is the name of plot
def plt_error(loss_hist_file, saved_direc, plot_name):
	loss_history = np.load(loss_hist_file) 
	loss_history = loss_history[()] #to deal with weird 0D array outputted by np.load()

	err = np.subtract(1, loss_history['acc'])
	val_err = np.subtract(1, loss_history['val_acc'])
	epoch = np.arange(1, len(err) + 1, 1)
	plt.plot(epoch, err)
	plt.plot(epoch, val_err)
	plt.title('Model Error')
	plt.xlabel('Epoch')
	plt.ylabel('Model Error')
	plt.legend(['training error','validation error'], loc='upper left')

	filename = saved_direc + plot_name + '.pdf'
	plt.savefig(filename, format = 'pdf')
