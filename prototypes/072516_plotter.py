import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl 
import matplotlib.lines as mlines
mpl.rcParams['pdf.fonttype'] = 42
import seaborn as sns

h01 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg0/2016-07-24_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg0_0.npz')['loss_history'][()]
h02 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg0/2016-07-24_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg0_1.npz')['loss_history'][()]
h03 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg0/2016-07-24_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg0_2.npz')['loss_history'][()]
h04 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg0/2016-07-24_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg0_3.npz')['loss_history'][()]
h05 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg0/2016-07-24_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg0_4.npz')['loss_history'][()]

h31 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg3/2016-07-24_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg3_0.npz')['loss_history'][()]
h32 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg3/2016-07-24_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg3_1.npz')['loss_history'][()]
h33 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg3/2016-07-24_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg3_2.npz')['loss_history'][()]
h34 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg3/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg3_3.npz')['loss_history'][()]
h35 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg3/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg3_4.npz')['loss_history'][()]

h41 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg4/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg4_0.npz')['loss_history'][()]
h42 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg4/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg4_1.npz')['loss_history'][()]
h43 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg4/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg4_2.npz')['loss_history'][()]
h44 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg4/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg4_3.npz')['loss_history'][()]
h45 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg4/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg4_4.npz')['loss_history'][()]

h51 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg5/2016-07-24_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg5_0.npz')['loss_history'][()]
h52 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg5/2016-07-24_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg5_1.npz')['loss_history'][()]
h53 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg5/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg5_2.npz')['loss_history'][()]
h54 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg5/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg5_3.npz')['loss_history'][()]
h55 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg5/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg5_4.npz')['loss_history'][()]

h61 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg6/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg6_0.npz')['loss_history'][()]
h62 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg6/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg6_1.npz')['loss_history'][()]
h63 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg6/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg6_2.npz')['loss_history'][()]
h64 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg6/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg6_3.npz')['loss_history'][()]
h65 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg6/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg6_4.npz')['loss_history'][()]

h71 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg7/2016-07-24_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg7_0.npz')['loss_history'][()]
h72 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg7/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg7_1.npz')['loss_history'][()]
h73 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg7/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg7_2.npz')['loss_history'][()]
h74 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg7/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg7_3.npz')['loss_history'][()]
h75 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/072516_networks/drop_reg7/2016-07-25_HeLa_set1_set5_61x61_feature_net_61x61_drop_reg7_4.npz')['loss_history'][()]

e01 = np.subtract(1, h01['acc'])
e02 = np.subtract(1, h02['acc'])
e03 = np.subtract(1, h03['acc'])
e04 = np.subtract(1, h04['acc'])
e05 = np.subtract(1, h05['acc'])
e0 = np.stack([e01, e02, e03, e04, e05], axis=0)

e31 = np.subtract(1, h31['acc'])
e32 = np.subtract(1, h32['acc'])
e33 = np.subtract(1, h33['acc'])
e34 = np.subtract(1, h34['acc'])
e35 = np.subtract(1, h35['acc'])
e3 = np.stack([e31, e32, e33, e34, e35], axis=0)

e41 = np.subtract(1, h41['acc'])
e42 = np.subtract(1, h42['acc'])
e43 = np.subtract(1, h43['acc'])
e44 = np.subtract(1, h44['acc'])
e45 = np.subtract(1, h45['acc'])
e4= np.stack([e41, e42, e43, e44, e45], axis=0)

e51 = np.subtract(1, h51['acc'])
e52 = np.subtract(1, h52['acc'])
e53 = np.subtract(1, h53['acc'])
e54 = np.subtract(1, h54['acc'])
e55 = np.subtract(1, h55['acc'])
e5 = np.stack([e51, e52, e53, e54, e55], axis=0)

e61 = np.subtract(1, h61['acc'])
e62 = np.subtract(1, h62['acc'])
e63 = np.subtract(1, h63['acc'])
e64 = np.subtract(1, h64['acc'])
e65 = np.subtract(1, h65['acc'])
e6 = np.stack([e61, e62, e63, e64, e65], axis=0)

e71 = np.subtract(1, h71['acc'])
e72 = np.subtract(1, h72['acc'])
e73 = np.subtract(1, h73['acc'])
e74 = np.subtract(1, h74['acc'])
e75 = np.subtract(1, h75['acc'])
e7 = np.stack([e71, e72, e73, e74, e75], axis=0)

v01 = np.subtract(1, h01['val_acc'])
v02 = np.subtract(1, h02['val_acc'])
v03 = np.subtract(1, h03['val_acc'])
v04 = np.subtract(1, h04['val_acc'])
v05 = np.subtract(1, h05['val_acc'])
v0 = np.stack([v01, v02, v03, v04, v05], axis=0)

v31 = np.subtract(1, h31['val_acc'])
v32 = np.subtract(1, h32['val_acc'])
v33 = np.subtract(1, h33['val_acc'])
v34 = np.subtract(1, h34['val_acc'])
v35 = np.subtract(1, h35['val_acc'])
v3 = np.stack([v31, v32, v33, v34, v35], axis=0)

v41 = np.subtract(1, h41['val_acc'])
v42 = np.subtract(1, h42['val_acc'])
v43 = np.subtract(1, h43['val_acc'])
v44 = np.subtract(1, h44['val_acc'])
v45 = np.subtract(1, h45['val_acc'])
v4 = np.stack([v41, v42, v43, v44, v45], axis=0)

v51 = np.subtract(1, h51['val_acc'])
v52 = np.subtract(1, h52['val_acc'])
v53 = np.subtract(1, h53['val_acc'])
v54 = np.subtract(1, h54['val_acc'])
v55 = np.subtract(1, h55['val_acc'])
v5 = np.stack([v51, v52, v53, v54, v55], axis=0)

v61 = np.subtract(1, h61['val_acc'])
v62 = np.subtract(1, h62['val_acc'])
v63 = np.subtract(1, h63['val_acc'])
v64 = np.subtract(1, h64['val_acc'])
v65 = np.subtract(1, h65['val_acc'])
v6 = np.stack([v61, v62, v63, v64, v65], axis=0)

v71 = np.subtract(1, h71['val_acc'])
v72 = np.subtract(1, h72['val_acc'])
v73 = np.subtract(1, h73['val_acc'])
v74 = np.subtract(1, h74['val_acc'])
v75 = np.subtract(1, h75['val_acc'])
v7 = np.stack([v71, v72, v73, v74, v75], axis=0)

emu0 = np.mean(e0, axis=0)
emu3 = np.mean(e3, axis=0)
emu4 = np.mean(e4, axis=0)
emu5 = np.mean(e5, axis=0)
emu6 = np.mean(e6, axis=0)
emu7 = np.mean(e7, axis=0)

vmu0 = np.mean(v0, axis=0)
vmu3 = np.mean(v3, axis=0)
vmu4 = np.mean(v4, axis=0)
vmu5 = np.mean(v5, axis=0)
vmu6 = np.mean(v6, axis=0)
vmu7 = np.mean(v7, axis=0)

es0 = np.std(e0, axis=0)
es3 = np.std(e3, axis=0)
es4 = np.std(e4, axis=0)
es5 = np.std(e5, axis=0)
es6 = np.std(e6, axis=0)
es7 = np.std(e7, axis=0)

vs0 = np.std(v0, axis=0)
vs3 = np.std(v3, axis=0)
vs4 = np.std(v4, axis=0)
vs5 = np.std(v5, axis=0)
vs6 = np.std(v6, axis=0)
vs7 = np.std(v7, axis=0)

epoch = np.arange(1, 26, 1)
sns.set_style("white")

solid = mlines.Line2D([], [], color='black', linestyle = '-', label = 'Training')
dashed = mlines.Line2D([],[], color='black', linestyle = '--', label= 'Validation')

plt.figure(0)
plt.plot(epoch, e01, 'k-', epoch, v01, 'k--', epoch, e02, 'k-', epoch, v02, 'k--', epoch, e03, 'k-', epoch, v03, 'k--', epoch, e04, 'k-', epoch, v04, 'k--', epoch, e05, 'k-', epoch, v05, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training and validation error: HeLa 61x61 reg=0')
plt.legend(handles=[solid,dashed])
filename = '/home/nquach/DeepCell2/prototypes/plots/072516_plots/HeLa_reg0.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(1)
plt.plot(epoch, e31, 'k-', epoch, v31, 'k--', epoch, e32, 'k-', epoch, v32, 'k--', epoch, e33, 'k-', epoch, v33, 'k--', epoch, e34, 'k-', epoch, v34, 'k--', epoch, e35, 'k-', epoch, v35, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training and validation error: HeLa 61x61 reg=1e-3')
plt.legend(handles=[solid,dashed])
filename = '/home/nquach/DeepCell2/prototypes/plots/072516_plots/HeLa_reg3.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(2)
plt.plot(epoch, e41, 'k-', epoch, v41, 'k--', epoch, e42, 'k-', epoch, v42, 'k--', epoch, e43, 'k-', epoch, v43, 'k--', epoch, e44, 'k-', epoch, v44, 'k--', epoch, e45, 'k-', epoch, v45, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training and validation error: HeLa 61x61 reg=1e-4')
plt.legend(handles=[solid,dashed])
filename = '/home/nquach/DeepCell2/prototypes/plots/072516_plots/HeLa_reg4.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(3)
plt.plot(epoch, e51, 'k-', epoch, v51, 'k--', epoch, e52, 'k-', epoch, v52, 'k--', epoch, e53, 'k-', epoch, v53, 'k--', epoch, e54, 'k-', epoch, v54, 'k--', epoch, e55, 'k-', epoch, v55, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training and validation error: HeLa 61x61 reg=1e-5')
plt.legend(handles=[solid,dashed])
filename = '/home/nquach/DeepCell2/prototypes/plots/072516_plots/HeLa_reg5.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(4)
plt.plot(epoch, e61, 'k-', epoch, v61, 'k--', epoch, e62, 'k-', epoch, v62, 'k--', epoch, e63, 'k-', epoch, v63, 'k--', epoch, e64, 'k-', epoch, v64, 'k--', epoch, e65, 'k-', epoch, v65, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training and validation error: HeLa 61x61 reg=1e-6')
plt.legend(handles=[solid,dashed])
filename = '/home/nquach/DeepCell2/prototypes/plots/072516_plots/HeLa_reg6.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(5)
plt.plot(epoch, e71, 'k-', epoch, v71, 'k--', epoch, e72, 'k-', epoch, v72, 'k--', epoch, e73, 'k-', epoch, v73, 'k--', epoch, e74, 'k-', epoch, v74, 'k--', epoch, e75, 'k-', epoch, v75, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training and validation error: HeLa 61x61 reg=1e-7')
plt.legend(handles=[solid,dashed])
filename = '/home/nquach/DeepCell2/prototypes/plots/072516_plots/HeLa_reg7.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(6)
plt.errorbar(epoch, emu0, yerr = es0, ls = '-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu0, yerr = vs0, ls = '--', color = (0.835,0.369,0))
plt.errorbar(epoch, emu5, yerr = es5, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu5, yerr = vs5, ls = '--', color=(0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Average training and validation error: reg=1e-5 vs reg=0')
plt.legend(['reg=0: training', 'reg=0: validation', 'reg=1e-5: training', 'reg=1e-5: validation'], loc='upper right')
filename = '/home/nquach/DeepCell2/prototypes/plots/072516_plots/HeLa_reg0_vs_reg5.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(7)
plt.errorbar(epoch, emu3, yerr = es3, ls = '-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu3, yerr = vs3, ls = '--', color = (0.835,0.369,0))
plt.errorbar(epoch, emu5, yerr = es5, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu5, yerr = vs5, ls = '--', color=(0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Average training and validation error: reg=1e-5 vs reg=1e-3')
plt.legend(['reg=1e-3: training', 'reg=1e-3: validation', 'reg=1e-5: training', 'reg=1e-5: validation'], loc='upper right')
filename = '/home/nquach/DeepCell2/prototypes/plots/072516_plots/HeLa_reg3_vs_reg5.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(8)
plt.errorbar(epoch, emu4, yerr = es4, ls = '-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu4, yerr = vs4, ls = '--', color = (0.835,0.369,0))
plt.errorbar(epoch, emu5, yerr = es5, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu5, yerr = vs5, ls = '--', color=(0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Average training and validation error: reg=1e-5 vs reg=1e-4')
plt.legend(['reg=1e-4: training', 'reg=1e-4: validation', 'reg=1e-5: training', 'reg=1e-5: validation'], loc='upper right')
filename = '/home/nquach/DeepCell2/prototypes/plots/072516_plots/HeLa_reg4_vs_reg5.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(9)
plt.errorbar(epoch, emu6, yerr = es6, ls = '-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu6, yerr = vs6, ls = '--', color = (0.835,0.369,0))
plt.errorbar(epoch, emu5, yerr = es5, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu5, yerr = vs5, ls = '--', color=(0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Average training and validation error: reg=1e-5 vs reg=1e-6')
plt.legend(['reg=1e-6: training', 'reg=1e-6: validation', 'reg=1e-5: training', 'reg=1e-5: validation'], loc='upper right')
filename = '/home/nquach/DeepCell2/prototypes/plots/072516_plots/HeLa_reg6_vs_reg5.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(10)
plt.errorbar(epoch, emu7, yerr = es7, ls = '-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu7, yerr = vs7, ls = '--', color = (0.835,0.369,0))
plt.errorbar(epoch, emu5, yerr = es5, ls = '-', color=(0,0.447,0.698))
plt.errorbar(epoch, vmu5, yerr = vs5, ls = '--', color=(0,0.447,0.698))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Average training and validation error: reg=1e-5 vs reg=1e-7')
plt.legend(['reg=1e-7: training', 'reg=1e-7: validation', 'reg=1e-5: training', 'reg=1e-5: validation'], loc='upper right')
filename = '/home/nquach/DeepCell2/prototypes/plots/072516_plots/HeLa_reg7_vs_reg5.pdf'
plt.savefig(filename, format='pdf')
plt.close()


