import numpy as np
import matplotlib.pyplot as plt
'''h1 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_1.npz')['loss_history'][()]
h2 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_2.npz')['loss_history'][()]
h3 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_3.npz')['loss_history'][()]
h4 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_4.npz')['loss_history'][()]
h5 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_5.npz')['loss_history'][()]
'''
h1 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm1_1.npz')['loss_history'][()]
#h2 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm2_2.npz')['loss_history'][()]
h3 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm3_3.npz')['loss_history'][()]
h4 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm4_4.npz')['loss_history'][()] 
h5 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm5_5.npz')['loss_history'][()]
e1 = np.subtract(1, h1['acc'])
#e2 = np.subtract(1, h2['acc'])
e3 = np.subtract(1, h3['acc'])
e4 = np.subtract(1, h4['acc'])
e5 = np.subtract(1, h5['acc'])

v1 = np.subtract(1, h1['val_acc'])
#v2 = np.subtract(1, h2['val_acc'])
v3 = np.subtract(1, h3['val_acc'])
v4 = np.subtract(1, h4['val_acc'])
v5 = np.subtract(1, h5['val_acc'])

epoch = np.arange(1, len(e1) + 1, 1)

plt.figure(0)
plt.plot(epoch, e1, 'b-', epoch, v1, 'b--', epoch, e3, 'r-', epoch, v3, 'r--', epoch, e4, 'k-', epoch, v4, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Model Error')
plt.title('Model Error of Batch Normalized Multiresolution Feature Net 61x61')
#plt.legend(['Model 1: training','Model 1: validation','Model 2: training','Model 2: validation', 'Model 3: training','Model 3: validation','Model 4: training','Model 4: validation','Model 5: training','Model 5: validation'], loc='upper right')
filename = '/home/nquach/DeepCell2/prototypes/plots/prelim_stdnorm.pdf'
plt.savefig(filename, format='pdf')

#epoch, e2, 'g-', epoch, v2, 'g--',