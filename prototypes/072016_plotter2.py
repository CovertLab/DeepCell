import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype'] = 42
import seaborn as sns

## --KEY FOR ENUMERATION-- ##
#1 feature_net_61x61_all
#2 feature_net_61x61_norm
#3 feature_net_61x61_dropout_all
#4 feature_net_61x61_dropout_norm
#5 bn_feature_net_61x61_all
#6 bn_feature_net_61x61_norm
#7 fn_multires_61x61_all
#8 fn_multires_61x61_norm
#9 feature_net_61x61_shear_all
#10 feature_net 61x61_shear_norm
#11 bn_multires_61x61_all
#12 bn_multires_61x61_norm

h11 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_1.npz')['loss_history'][()]
h12 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_2.npz')['loss_history'][()]
h13 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_3.npz')['loss_history'][()]

h31 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_dropout_1.npz')['loss_history'][()]
h32 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_dropout_2.npz')['loss_history'][()]
h33 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_dropout_3.npz')['loss_history'][()]

h51 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-12_HeLa_all_61x61_bn_feature_net_61x61_1.npz')['loss_history'][()]
h52 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-12_HeLa_all_61x61_bn_feature_net_61x61_2.npz')['loss_history'][()]
h53 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-14_HeLa_all_61x61_bn_feature_net_61x61_3.npz')['loss_history'][()]

h71 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_fn_multires_61x61_1.npz')['loss_history'][()]
h72 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_fn_multires_61x61_2.npz')['loss_history'][()]
h73 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_fn_multires_61x61_3.npz')['loss_history'][()]

h111 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_bn_multires_61x61_1.npz')['loss_history'][()]
h112 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_bn_multires_61x61_2.npz')['loss_history'][()]
h113 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_bn_multires_61x61_3.npz')['loss_history'][()]

e11 = np.subtract(1, h11['acc'])
e12 = np.subtract(1, h12['acc'])
e13 = np.subtract(1, h13['acc'])
e1 = np.stack([e11, e12, e13], axis=0)
print e1.shape

e31 = np.subtract(1, h31['acc'])
e32 = np.subtract(1, h32['acc'])
e33 = np.subtract(1, h33['acc'])
e3 = np.stack([e31, e32, e33], axis=0)

e51 = np.subtract(1, h51['acc'])
e52 = np.subtract(1, h52['acc'])
e53 = np.subtract(1, h53['acc'])
e5 = np.stack([e51, e52, e53], axis=0)

e71 = np.subtract(1, h71['acc'])
e72 = np.subtract(1, h72['acc'])
e73 = np.subtract(1, h73['acc'])
e7 = np.stack([e71, e72, e73], axis=0)

e111 = np.subtract(1, h111['acc'])
e112 = np.subtract(1, h112['acc'])
e113 = np.subtract(1, h113['acc'])
e11 = np.stack([e111, e112, e113], axis=0)

v11 = np.subtract(1, h11['val_acc'])
v12 = np.subtract(1, h12['val_acc'])
v13 = np.subtract(1, h13['val_acc'])
v1 = np.stack([v11, v12, v13], axis=0)
print v1.shape

v31 = np.subtract(1, h31['val_acc'])
v32 = np.subtract(1, h32['val_acc'])
v33 = np.subtract(1, h33['val_acc'])
v3 = np.stack([v31, v32, v33], axis=0)

v51 = np.subtract(1, h51['val_acc'])
v52 = np.subtract(1, h52['val_acc'])
v53 = np.subtract(1, h53['val_acc'])
v5 = np.stack([v51, v52, v53], axis=0)

v71 = np.subtract(1, h71['val_acc'])
v72 = np.subtract(1, h72['val_acc'])
v73 = np.subtract(1, h73['val_acc'])
v7 = np.stack([v71, v72, v73], axis=0)

v111 = np.subtract(1, h111['val_acc'])
v112 = np.subtract(1, h112['val_acc'])
v113 = np.subtract(1, h113['val_acc'])
v11 = np.stack([v111, v112, v113], axis=0)

emu1 = np.mean(e1, axis=0)
emu3 = np.mean(e3, axis=0)
emu5 = np.mean(e5, axis=0)
emu7 = np.mean(e7, axis=0)
emu11 = np.mean(e11, axis=0)
print emu1.shape

vmu1 = np.mean(v1, axis=0)
vmu3 = np.mean(v3, axis=0)
vmu5 = np.mean(v5, axis=0)
vmu7 = np.mean(v7, axis=0)
vmu11 = np.mean(v11, axis=0)
print vmu1.shape

es1 = np.std(e1, axis=0)
es3 = np.std(e3, axis=0)
es5 = np.std(e5, axis=0)
es7 = np.std(e7, axis=0)
es11 = np.std(e11, axis=0)
print es1.shape

vs1 = np.std(v1, axis=0)
vs3 = np.std(v3, axis=0)
vs5 = np.std(v5, axis=0)
vs7 = np.std(v7, axis=0)
vs11 = np.std(v11, axis=0)
print vs1.shape

epoch = np.arange(1, 26, 1)
print len(epoch)
print len(emu1)
sns.set_style("white")

plt.figure(0)
plt.errorbar(epoch, emu1, yerr = es1, ls='-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu1, yerr= vs1, ls='--', color=(0.835,0.369,0))
plt.errorbar(epoch, emu3, yerr = es3, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu3, yerr = vs3, ls = '--', color = (0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(['No Dropout: training','No Dropout: validation', 'Dropout: training','Dropout: validation'], loc='upper right')
plt.title('Average training and validation error: Dropout vs No Dropout')
filename = '/home/nquach/DeepCell2/prototypes/plots/072016_plots/vanilla_vs_dropout.pdf'
plt.savefig(filename, facecolor = 'w', format = 'pdf')
plt.close()

plt.figure(1)
plt.errorbar(epoch, emu1, yerr = es1, ls='-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu1, yerr=vs1, ls='--', color=(0.835,0.369,0))
plt.errorbar(epoch, emu5, yerr = es5, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu5, yerr = vs5, ls = '--', color = (0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(['No batch norm: training','No batch norm: validation', 'Batch norm: training','Batch norm: validation'], loc='upper right')
plt.title('Average training and validation error: Batch Normalization vs No Batch Normalization')
filename = '/home/nquach/DeepCell2/prototypes/plots/072016_plots/vanilla_vs_bn.pdf'
plt.savefig(filename, format = 'pdf')
plt.close()

plt.figure(2)
plt.errorbar(epoch, emu1, yerr = es1, ls='-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu1, yerr=vs1, ls='--', color=(0.835,0.369,0))
plt.errorbar(epoch, emu7, yerr = es7, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu7, yerr = vs7, ls = '--', color = (0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(['No multiresolution: training','No multiresolution: validation', 'Multiresolution: training','Multiresolution: validation'], loc='upper right')
plt.title('Average training and validation error: Multiresolution vs No Multiresolution')
filename = '/home/nquach/DeepCell2/prototypes/plots/072016_plots/vanilla_vs_multires.pdf'
plt.savefig(filename, format = 'pdf')
plt.close()

plt.figure(3)
plt.errorbar(epoch, emu1, yerr = es1, ls='-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu1, yerr=vs1, ls='--', color=(0.835,0.369,0))
plt.errorbar(epoch, emu11, yerr = es11, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu11, yerr = vs11, ls = '--', color = (0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(['No batch normalized multires: training','No batch normalized multires: validation', 'Batch normalized multires: training','Batch normalized multires: validation'], loc='upper right')
plt.title('Average training and validation error: Batch normalized multires vs No batch normalized multires')
filename = '/home/nquach/DeepCell2/prototypes/plots/072016_plots/vanilla_vs_bn_multires.pdf'
plt.savefig(filename, format = 'pdf')
plt.close()






 