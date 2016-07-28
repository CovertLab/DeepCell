import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl 
import matplotlib.lines as mlines
mpl.rcParams['pdf.fonttype'] = 42
import seaborn as sns

h11 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-20_HeLa_all_61x61_feature_net_61x61_reg0_0.npz')['loss_history'][()]
h12 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-20_HeLa_all_61x61_feature_net_61x61_reg0_1.npz')['loss_history'][()]
h13 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-20_HeLa_all_61x61_feature_net_61x61_reg0_2.npz')['loss_history'][()]

h21 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-20_HeLa_all_61x61_feature_net_61x61_reg5_0.npz')['loss_history'][()]
h22 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-20_HeLa_all_61x61_feature_net_61x61_reg5_1.npz')['loss_history'][()]
h23 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-20_HeLa_all_61x61_feature_net_61x61_reg5_2.npz')['loss_history'][()]

h31 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-20_HeLa_all_61x61_feature_net_61x61_reg6_0.npz')['loss_history'][()]
h32 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-21_HeLa_all_61x61_feature_net_61x61_reg6_1.npz')['loss_history'][()]
h33 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-20_HeLa_all_61x61_feature_net_61x61_reg6_2.npz')['loss_history'][()]

h41 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-20_HeLa_all_61x61_feature_net_61x61_reg7_0.npz')['loss_history'][()]
h42 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-20_HeLa_all_61x61_feature_net_61x61_reg7_1.npz')['loss_history'][()]
h43 = np.load('/home/nquach/DeepCell2/trained_networks/Experimental/HeLa/HeLa_reg_tests/2016-07-20_HeLa_all_61x61_feature_net_61x61_reg7_2.npz')['loss_history'][()]

e11 = np.subtract(1, h11['acc'])
e12 = np.subtract(1, h12['acc'])
e13 = np.subtract(1, h13['acc'])
e1 = np.stack([e11, e12, e13], axis=0)

e21 = np.subtract(1, h21['acc'])
e22 = np.subtract(1, h22['acc'])
e23 = np.subtract(1, h23['acc'])
e2 = np.stack([e21, e22, e23], axis=0)

e31 = np.subtract(1, h31['acc'])
e32 = np.subtract(1, h32['acc'])
e33 = np.subtract(1, h33['acc'])
e3 = np.stack([e31, e32, e33], axis=0)

e41 = np.subtract(1, h41['acc'])
e42 = np.subtract(1, h42['acc'])
e43 = np.subtract(1, h43['acc'])
e4 = np.stack([e41, e42, e43], axis=0)

v11 = np.subtract(1, h11['val_acc'])
v12 = np.subtract(1, h12['val_acc'])
v13 = np.subtract(1, h13['val_acc'])
v1 = np.stack([v11, v12, v13], axis=0)

v21 = np.subtract(1, h21['val_acc'])
v22 = np.subtract(1, h22['val_acc'])
v23 = np.subtract(1, h23['val_acc'])
v2 = np.stack([v21, v22, v23], axis=0)

v31 = np.subtract(1, h31['val_acc'])
v32 = np.subtract(1, h32['val_acc'])
v33 = np.subtract(1, h33['val_acc'])
v3 = np.stack([v31, v32, v33], axis=0)

v41 = np.subtract(1, h41['val_acc'])
v42 = np.subtract(1, h42['val_acc'])
v43 = np.subtract(1, h43['val_acc'])
v4 = np.stack([v41, v42, v43], axis=0)

emu1 = np.mean(e1, axis=0)
emu2 = np.mean(e2, axis=0)
emu3 = np.mean(e3, axis=0)
emu4 = np.mean(e4, axis=0)

vmu1 = np.mean(v1, axis=0)
vmu2 = np.mean(v2, axis=0)
vmu3 = np.mean(v3, axis=0)
vmu4 = np.mean(v4, axis=0)

es1 = np.std(e1, axis=0)
es2 = np.std(e2, axis=0)
es3 = np.std(e3, axis=0)
es4 = np.std(e4, axis=0)

vs1 = np.std(v1, axis=0)
vs2 = np.std(v2, axis=0)
vs3 = np.std(v3, axis=0)
vs4 = np.std(v4, axis=0)

epoch = np.arange(1, 26, 1)
sns.set_style("white")

plt.figure(0)
plt.errorbar(epoch, emu2, yerr = es2, ls='-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu2, yerr= vs2, ls='--', color=(0.835,0.369,0))
plt.errorbar(epoch, emu1, yerr = es1, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu1, yerr = vs1, ls = '--', color =(0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(['reg = 1e-5 : training','reg = 1e-5 : validation', 'reg = 0 : training','reg = 0 : validation'], loc='upper right')
plt.title('Average training and validation error: reg=1e-5 vs reg=0')
filename = '/home/nquach/DeepCell2/prototypes/plots/072116_plots/reg5_vs_reg0.pdf'
plt.savefig(filename, facecolor = 'w', format = 'pdf')
plt.close()

plt.figure(1)
plt.errorbar(epoch, emu2, yerr = es2, ls='-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu2, yerr= vs2, ls='--', color=(0.835,0.369,0))
plt.errorbar(epoch, emu3, yerr = es3, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu3, yerr = vs3, ls = '--', color =(0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(['reg = 1e-5 : training','reg = 1e-5 : validation', 'reg = 1e-6 : training','reg = 1e-6 : validation'], loc='upper right')
plt.title('Average training and validation error: reg=1e-5 vs reg=1e-6')
filename = '/home/nquach/DeepCell2/prototypes/plots/072116_plots/reg5_vs_reg6.pdf'
plt.savefig(filename, facecolor = 'w', format = 'pdf')
plt.close()

plt.figure(3)
plt.errorbar(epoch, emu2, yerr = es2, ls='-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu2, yerr= vs2, ls='--', color=(0.835,0.369,0))
plt.errorbar(epoch, emu4, yerr = es4, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu4, yerr = vs4, ls = '--', color =(0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(['reg = 1e-5 : training','reg = 1e-5 : validation', 'reg = 1e-7 : training','reg = 1e-7 : validation'], loc='upper right')
plt.title('Average training and validation error: reg=1e-5 vs reg=1e-7')
filename = '/home/nquach/DeepCell2/prototypes/plots/072116_plots/reg5_vs_reg7.pdf'
plt.savefig(filename, facecolor = 'w', format = 'pdf')
plt.close()

solid = mlines.Line2D([], [], color='black', linestyle = '-', label = 'Training')
dashed = mlines.Line2D([],[], color='black', linestyle = '--', label= 'Validation')

plt.figure(4)
plt.plot(epoch, e11, 'k-', epoch, v11, 'k--', epoch, e12, 'k-', epoch, v12, 'k--', epoch, e13, 'k-', epoch, v13, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(handles=[solid,dashed])
plt.title('Training and validation classification error: reg = 0')
filename = '/home/nquach/DeepCell2/prototypes/plots/072116_plots/reg0.pdf'
plt.savefig(filename, facecolor = 'w', format = 'pdf')
plt.close()

plt.figure(5)
plt.plot(epoch, e21, 'k-', epoch, v21, 'k--', epoch, e22, 'k-', epoch, v22, 'k--', epoch, e23, 'k-', epoch, v23, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(handles=[solid,dashed])
plt.title('Training and validation classification error: reg = 1e-5')
filename = '/home/nquach/DeepCell2/prototypes/plots/072116_plots/reg5.pdf'
plt.savefig(filename, facecolor = 'w', format = 'pdf')
plt.close()

plt.figure(6)
plt.plot(epoch, e31, 'k-', epoch, v31, 'k--', epoch, e32, 'k-', epoch, v32, 'k--', epoch, e33, 'k-', epoch, v33, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(handles=[solid,dashed])
plt.title('Training and validation classification error: reg = 1e-6')
filename = '/home/nquach/DeepCell2/prototypes/plots/072116_plots/reg6.pdf'
plt.savefig(filename, facecolor = 'w', format = 'pdf')
plt.close()

plt.figure(7)
plt.plot(epoch, e41, 'k-', epoch, v41, 'k--', epoch, e42, 'k-', epoch, v42, 'k--', epoch, e43, 'k-', epoch, v43, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(handles=[solid,dashed])
plt.title('Training and validation classification error: reg = 1e-7')
filename = '/home/nquach/DeepCell2/prototypes/plots/072116_plots/reg7.pdf'
plt.savefig(filename, facecolor = 'w', format = 'pdf')
plt.close()




