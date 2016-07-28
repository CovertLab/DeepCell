import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

h11 = np.load('/home/nquach/DeepCell2/trained_networks/RAW40X/2016-07-13_RAW40X_all_61x61_bn_feature_net_61x61_raw_0.npz')['loss_history'][()]
h12 = np.load('/home/nquach/DeepCell2/trained_networks/RAW40X/2016-07-13_RAW40X_all_61x61_bn_feature_net_61x61_raw_1.npz')['loss_history'][()]
h13 = np.load('/home/nquach/DeepCell2/trained_networks/RAW40X/2016-07-13_RAW40X_all_61x61_bn_feature_net_61x61_raw_2.npz')['loss_history'][()]
h14 = np.load('/home/nquach/DeepCell2/trained_networks/RAW40X/2016-07-13_RAW40X_all_61x61_bn_feature_net_61x61_raw_3.npz')['loss_history'][()]
h15 = np.load('/home/nquach/DeepCell2/trained_networks/RAW40X/2016-07-13_RAW40X_all_61x61_bn_feature_net_61x61_raw_4.npz')['loss_history'][()]

h21 = np.load('/home/nquach/DeepCell2/trained_networks/MCF10A_3T3_semantic/2016-07-13_MCF10A_3T3_semantic_61x61_bn_feature_net_61x61_semantic_0.npz')['loss_history'][()]
h22 = np.load('/home/nquach/DeepCell2/trained_networks/MCF10A_3T3_semantic/2016-07-13_MCF10A_3T3_semantic_61x61_bn_feature_net_61x61_semantic_1.npz')['loss_history'][()]
h23 = np.load('/home/nquach/DeepCell2/trained_networks/MCF10A_3T3_semantic/2016-07-13_MCF10A_3T3_semantic_61x61_bn_feature_net_61x61_semantic_2.npz')['loss_history'][()]
h24 = np.load('/home/nquach/DeepCell2/trained_networks/MCF10A_3T3_semantic/2016-07-13_MCF10A_3T3_semantic_61x61_bn_feature_net_61x61_semantic_3.npz')['loss_history'][()]
h25 = np.load('/home/nquach/DeepCell2/trained_networks/MCF10A_3T3_semantic/2016-07-13_MCF10A_3T3_semantic_61x61_bn_feature_net_61x61_semantic_4.npz')['loss_history'][()]

h31 = np.load('/home/nquach/DeepCell2/trained_networks/BMDM/2016-07-13_BMDM_61x61_bn_feature_net_61x61_BMDM_0.npz')['loss_history'][()]
h32 = np.load('/home/nquach/DeepCell2/trained_networks/BMDM/2016-07-13_BMDM_61x61_bn_feature_net_61x61_BMDM_1.npz')['loss_history'][()]
h33 = np.load('/home/nquach/DeepCell2/trained_networks/BMDM/2016-07-13_BMDM_61x61_bn_feature_net_61x61_BMDM_2.npz')['loss_history'][()]
h34 = np.load('/home/nquach/DeepCell2/trained_networks/BMDM/2016-07-15_BMDM_61x61_bn_feature_net_61x61_BMDM_3.npz')['loss_history'][()]
h35 = np.load('/home/nquach/DeepCell2/trained_networks/BMDM/2016-07-15_BMDM_61x61_bn_feature_net_61x61_BMDM_4.npz')['loss_history'][()]

h41 = np.load('/home/nquach/DeepCell2/trained_networks/Nuclear/2016-07-12_nuclei_all_61x61_bn_feature_net_61x61_0.npz')['loss_history'][()]
h42 = np.load('/home/nquach/DeepCell2/trained_networks/Nuclear/2016-07-12_nuclei_all_61x61_bn_feature_net_61x61_1.npz')['loss_history'][()]
h43 = np.load('/home/nquach/DeepCell2/trained_networks/Nuclear/2016-07-12_nuclei_all_61x61_bn_feature_net_61x61_2.npz')['loss_history'][()]
h44 = np.load('/home/nquach/DeepCell2/trained_networks/Nuclear/2016-07-12_nuclei_all_61x61_bn_feature_net_61x61_3.npz')['loss_history'][()]
h45 = np.load('/home/nquach/DeepCell2/trained_networks/Nuclear/2016-07-12_nuclei_all_61x61_bn_feature_net_61x61_4.npz')['loss_history'][()]

h51 = np.load('/home/nquach/DeepCell2/trained_networks/3T3/2016-07-12_3T3_all_61x61_bn_feature_net_61x61_0.npz')['loss_history'][()]
h52 = np.load('/home/nquach/DeepCell2/trained_networks/3T3/2016-07-12_3T3_all_61x61_bn_feature_net_61x61_1.npz')['loss_history'][()]
h53 = np.load('/home/nquach/DeepCell2/trained_networks/3T3/2016-07-12_3T3_all_61x61_bn_feature_net_61x61_2.npz')['loss_history'][()]
h54 = np.load('/home/nquach/DeepCell2/trained_networks/3T3/2016-07-12_3T3_all_61x61_bn_feature_net_61x61_3.npz')['loss_history'][()]
h55 = np.load('/home/nquach/DeepCell2/trained_networks/3T3/2016-07-14_3T3_all_61x61_bn_feature_net_61x61_4.npz')['loss_history'][()]

h61 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-12_HeLa_all_61x61_bn_feature_net_61x61_0.npz')['loss_history'][()]
h62 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-12_HeLa_all_61x61_bn_feature_net_61x61_1.npz')['loss_history'][()]
h63 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-12_HeLa_all_61x61_bn_feature_net_61x61_2.npz')['loss_history'][()]
h64 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-14_HeLa_all_61x61_bn_feature_net_61x61_3.npz')['loss_history'][()]
h65 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-14_HeLa_all_61x61_bn_feature_net_61x61_4.npz')['loss_history'][()]

e11 = np.subtract(1, h11['acc'])
e12 = np.subtract(1, h12['acc'])
e13 = np.subtract(1, h13['acc'])
e14 = np.subtract(1, h14['acc'])
e15 = np.subtract(1, h15['acc'])

e21 = np.subtract(1, h21['acc'])
e22 = np.subtract(1, h22['acc'])
e23 = np.subtract(1, h23['acc'])
e24 = np.subtract(1, h24['acc'])
e25 = np.subtract(1, h25['acc'])

e31 = np.subtract(1, h31['acc'])
e32 = np.subtract(1, h32['acc'])
e33 = np.subtract(1, h33['acc'])
e34 = np.subtract(1, h34['acc'])
e35 = np.subtract(1, h35['acc'])

e41 = np.subtract(1, h41['acc'])
e42 = np.subtract(1, h42['acc'])
e43 = np.subtract(1, h43['acc'])
e44 = np.subtract(1, h44['acc'])
e45 = np.subtract(1, h45['acc'])

e51 = np.subtract(1, h51['acc'])
e52 = np.subtract(1, h52['acc'])
e53 = np.subtract(1, h53['acc'])
e54 = np.subtract(1, h54['acc'])
e55 = np.subtract(1, h55['acc'])

e61 = np.subtract(1, h61['acc'])
e62 = np.subtract(1, h62['acc'])
e63 = np.subtract(1, h63['acc'])
e64 = np.subtract(1, h64['acc'])
e65 = np.subtract(1, h65['acc'])

v11 = np.subtract(1, h11['val_acc'])
v12 = np.subtract(1, h12['val_acc'])
v13 = np.subtract(1, h13['val_acc'])
v14 = np.subtract(1, h14['val_acc'])
v15 = np.subtract(1, h15['val_acc'])

v21 = np.subtract(1, h21['val_acc'])
v22 = np.subtract(1, h22['val_acc'])
v23 = np.subtract(1, h23['val_acc'])
v24 = np.subtract(1, h24['val_acc'])
v25 = np.subtract(1, h25['val_acc'])

v31 = np.subtract(1, h31['val_acc'])
v32 = np.subtract(1, h32['val_acc'])
v33 = np.subtract(1, h33['val_acc'])
v34 = np.subtract(1, h34['val_acc'])
v35 = np.subtract(1, h35['val_acc'])

v41 = np.subtract(1, h41['val_acc'])
v42 = np.subtract(1, h42['val_acc'])
v43 = np.subtract(1, h43['val_acc'])
v44 = np.subtract(1, h44['val_acc'])
v45 = np.subtract(1, h45['val_acc'])

v51 = np.subtract(1, h51['val_acc'])
v52 = np.subtract(1, h52['val_acc'])
v53 = np.subtract(1, h53['val_acc'])
v54 = np.subtract(1, h54['val_acc'])
v55 = np.subtract(1, h55['val_acc'])

v61 = np.subtract(1, h61['val_acc'])
v62 = np.subtract(1, h62['val_acc'])
v63 = np.subtract(1, h63['val_acc'])
v64 = np.subtract(1, h64['val_acc'])
v65 = np.subtract(1, h65['val_acc'])

#data1 = [(e11, e12, e13, e14, e15), (v11, v12, v13, v14, v15), (e21, e22, e23, e24, e25), (v21, v22, v23, v24, v25), (e31, e32, e33, e34, e35), (v31, v32, v33, v34, v35), (e41, e42, e43, e44, e45), (v41, v42, v43, v44, v45), (e51, e52, e53, e54, e55), (v51, v52, v53, v54, v55), (e61, e62, e63, e64, e65), (v61, v62, v63, v64, v65)]
#data1 = np.moveaxis(np.asarray(data1), 0, -1)

data2 = [(e11, e12, e13, e14, e15), (v11, v12, v13, v14, v15), (e21, e22, e23, e24, e25), (v21, v22, v23, v24, v25), (e31, e32, e33, e34, e35), (v31, v32, v33, v34, v35)]
data2 = np.moveaxis(np.asarray(data2), 0, -1)

data3 = [(e41, e42, e43, e44, e45), (v41, v42, v43, v44, v45), (e51, e52, e53, e54, e55), (v51, v52, v53, v54, v55), (e61, e62, e63, e64, e65), (v61, v62, v63, v64, v65)]
data3 = np.moveaxis(np.asarray(data3), 0, -1)

palette = ["#8c510a","#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e", "#c51b7d", "#e9a3c9", "#fde0ef", "#e6f5d0", "#a1d76a", "#4d9221"]
blind_palette = [(0,0,0), (0.902,0.624,0), (0.337,0.706,0.914), (0,0.620,0.451), (0.941, 0.894, 0.259),(0,0.447,0.698), (0.835,0.369,0), (0.800,0.475,0.655)]
blind_paired_palette = [(0.902,0.624,0), (1, 0.718, 0), (0.337,0.706,0.914), (0.388, 0.812, 1), (0,0.620,0.451), (0,0.713, 0.519), (0,0.447,0.698), (0, 0.514, 0.803)]
sns.set_style("white")
#sns.set_palette(palette)
sns.set_palette("Paired")
ax_lim = [0, 25, 0.05, 0.30]
plt.figure(0)
#sns.tsplot(data=data1, condition=['RAW40X: training', 'RAW40X: validation', 'MCF10A/3T3 semantic: training','MCF10A/3T3 semantic: validation', 'BMDM: training', 'BMDM: validation', 'nuclei: training', 'nuclei: validation', '3T3: training', '3T3: validation', 'HeLa: training', 'HeLa: validation'], err_style='ci_bars')
sns.tsplot(data=data2, condition=['RAW40X: training', 'RAW40X: validation', 'MCF10A/3T3 semantic: training','MCF10A/3T3 semantic: validation', 'BMDM: training', 'BMDM: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: bn_feature_net_61x61')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071316_plots/bn_feature_net_61x61_blind_paired1.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(1)
sns.tsplot(data=data3, condition=['nuclei: training', 'nuclei: validation', '3T3: training', '3T3: validation', 'HeLa: training', 'HeLa: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: bn_feature_net_61x61')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071316_plots/bn_feature_net_61x61_blind_paired2.pdf'
plt.savefig(filename, format='pdf')
plt.close()
