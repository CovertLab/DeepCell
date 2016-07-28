import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
mpl.rcParams['pdf.fonttype'] = 42

h11 = np.load('/home/nquach/DeepCell2/trained_networks/3T3/2016-07-12_3T3_all_61x61_bn_feature_net_61x61_0.npz')['loss_history'][()]
h12 = np.load('/home/nquach/DeepCell2/trained_networks/3T3/2016-07-12_3T3_all_61x61_bn_feature_net_61x61_1.npz')['loss_history'][()]
h13 = np.load('/home/nquach/DeepCell2/trained_networks/3T3/2016-07-12_3T3_all_61x61_bn_feature_net_61x61_2.npz')['loss_history'][()]
h14 = np.load('/home/nquach/DeepCell2/trained_networks/3T3/2016-07-12_3T3_all_61x61_bn_feature_net_61x61_3.npz')['loss_history'][()]
h15 = np.load('/home/nquach/DeepCell2/trained_networks/3T3/2016-07-14_3T3_all_61x61_bn_feature_net_61x61_4.npz')['loss_history'][()]

h21 = np.load('/home/nquach/DeepCell2/trained_networks/MCF10A/2016-07-11_MCF10A_61x61_bn_feature_net_61x61_0.npz')['loss_history'][()]
h22 = np.load('/home/nquach/DeepCell2/trained_networks/MCF10A/2016-07-11_MCF10A_61x61_bn_feature_net_61x61_1.npz')['loss_history'][()]
h23 = np.load('/home/nquach/DeepCell2/trained_networks/MCF10A/2016-07-11_MCF10A_61x61_bn_feature_net_61x61_2.npz')['loss_history'][()]
h24 = np.load('/home/nquach/DeepCell2/trained_networks/MCF10A/2016-07-11_MCF10A_61x61_bn_feature_net_61x61_3.npz')['loss_history'][()]
h25 = np.load('/home/nquach/DeepCell2/trained_networks/MCF10A/2016-07-11_MCF10A_61x61_bn_feature_net_61x61_4.npz')['loss_history'][()]

h31 = np.load('/home/nquach/DeepCell2/trained_networks/RAW40X/2016-07-13_RAW40X_all_61x61_bn_feature_net_61x61_raw_0.npz')['loss_history'][()]
h32 = np.load('/home/nquach/DeepCell2/trained_networks/RAW40X/2016-07-13_RAW40X_all_61x61_bn_feature_net_61x61_raw_1.npz')['loss_history'][()]
h33 = np.load('/home/nquach/DeepCell2/trained_networks/RAW40X/2016-07-13_RAW40X_all_61x61_bn_feature_net_61x61_raw_2.npz')['loss_history'][()]
h34 = np.load('/home/nquach/DeepCell2/trained_networks/RAW40X/2016-07-13_RAW40X_all_61x61_bn_feature_net_61x61_raw_3.npz')['loss_history'][()]
h35 = np.load('/home/nquach/DeepCell2/trained_networks/RAW40X/2016-07-13_RAW40X_all_61x61_bn_feature_net_61x61_raw_4.npz')['loss_history'][()]

h41 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-12_HeLa_all_61x61_bn_feature_net_61x61_0.npz')['loss_history'][()]
h42 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-12_HeLa_all_61x61_bn_feature_net_61x61_1.npz')['loss_history'][()]
h43 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-12_HeLa_all_61x61_bn_feature_net_61x61_2.npz')['loss_history'][()]
h44 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-14_HeLa_all_61x61_bn_feature_net_61x61_3.npz')['loss_history'][()]
h45 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-14_HeLa_all_61x61_bn_feature_net_61x61_4.npz')['loss_history'][()]

h51 = np.load('/home/nquach/DeepCell2/trained_networks/BMDM/2016-07-13_BMDM_61x61_bn_feature_net_61x61_BMDM_0.npz')['loss_history'][()]
h52 = np.load('/home/nquach/DeepCell2/trained_networks/BMDM/2016-07-13_BMDM_61x61_bn_feature_net_61x61_BMDM_1.npz')['loss_history'][()]
h53 = np.load('/home/nquach/DeepCell2/trained_networks/BMDM/2016-07-13_BMDM_61x61_bn_feature_net_61x61_BMDM_2.npz')['loss_history'][()]
h54 = np.load('/home/nquach/DeepCell2/trained_networks/BMDM/2016-07-15_BMDM_61x61_bn_feature_net_61x61_BMDM_3.npz')['loss_history'][()]
h55 = np.load('/home/nquach/DeepCell2/trained_networks/BMDM/2016-07-15_BMDM_61x61_bn_feature_net_61x61_BMDM_4.npz')['loss_history'][()]

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


epoch = np.arange(1, len(e11) + 1, 1)
'''
def plot(training, validation, cell_type, model, i):
	plt.figure(i)
	plt.plot(epoch, training, 'k-', epoch, validation, 'k--')
	plt.title('Training and validation classification error: ' + cell_type)
	plt.xlabel('Epoch')
	plt.ylabel('Error')	
	plt.legend(['Training','Validation'], loc='upper right')
	filename = r'/home/nquach/DeepCell2/prototypes/plots/072016_plots/' + cell_type + '_' + str(model) + '.pdf'
	plt.savefig(filename, format='pdf')
	plt.close()

types = ['3T3','MCF10A','RAW40X','HeLa','BMDM']
data1 = [(e11, v11), (e12, v12), (e13, v13), (e14, v14), (e15, v15)]
data2 = [(e21, v21), (e22, v22), (e23, v23), (e24, v24), (e25, v25)]
data3 = [(e31, v31), (e32, v32), (e33, v33), (e34, v34), (e35, v35)]
data4 = [(e41, v41), (e42, v42), (e43, v43), (e44, v44), (e45, v45)]
data5 = [(e51, v51), (e52, v52), (e53, v53), (e54, v54), (e55, v55)]

all_data = [data1, data2, data3, data4, data5]

i = 0
for cell in types:
	for data in all_data:
		model = 0
		for pair in data:
			plot(pair[0], pair[1], cell, model, i)
			model += 1
			i += 1

'''
solid = mlines.Line2D([], [], color='black', linestyle = '-', label = 'Training')
dashed = mlines.Line2D([],[], color='black', linestyle = '--', label= 'Validation')

plt.figure(0)
plt.plot(epoch, e11, 'k-',epoch, e12, 'k-', epoch, e13, 'k-', epoch, e14, 'k-', epoch, e15, 'k-', epoch, v11, 'k--', epoch, v12, 'k--', epoch, v13, 'k--', epoch, v14, 'k--', epoch, v15, 'k--')
plt.title('Training and validation classification error: 3T3')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(handles=[solid,dashed])
filename = '/home/nquach/DeepCell2/prototypes/plots/072016_plots/3T3_all.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(1)
plt.plot(epoch, e21, 'k-',epoch, e22, 'k-', epoch, e23, 'k-', epoch, e24, 'k-', epoch, e25, 'k-', epoch, v21, 'k--', epoch, v22, 'k--', epoch, v23, 'k--', epoch, v24, 'k--', epoch, v25, 'k--')
plt.title('Training and validation classification error: MCF10A')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(handles=[solid,dashed])
filename = '/home/nquach/DeepCell2/prototypes/plots/072016_plots/MCF10A_all.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(2)
plt.plot(epoch, e31, 'k-',epoch, e32, 'k-', epoch, e33, 'k-', epoch, e34, 'k-', epoch, e35, 'k-', epoch, v31, 'k--', epoch, v32, 'k--', epoch, v33, 'k--', epoch, v34, 'k--', epoch, v35, 'k--')
plt.title('Training and validation classification error: RAW40X')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(handles=[solid,dashed])
filename = '/home/nquach/DeepCell2/prototypes/plots/072016_plots/RAW40X_all.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(3)
plt.plot(epoch, e41, 'k-',epoch, e42, 'k-', epoch, e43, 'k-', epoch, e44, 'k-', epoch, e45, 'k-', epoch, v41, 'k--', epoch, v42, 'k--', epoch, v43, 'k--', epoch, v44, 'k--', epoch, v45, 'k--')
plt.title('Training and validation classification error: HeLa')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(handles=[solid,dashed])
filename = '/home/nquach/DeepCell2/prototypes/plots/072016_plots/HeLa_all.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(4)
plt.plot(epoch, e51, 'k-',epoch, e52, 'k-', epoch, e53, 'k-', epoch, e54, 'k-', epoch, e55, 'k-', epoch, v51, 'k--', epoch, v52, 'k--', epoch, v53, 'k--', epoch, v54, 'k--', epoch, v55, 'k--')
plt.title('Training and validation classification error: BMDM')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(handles=[solid,dashed])
filename = '/home/nquach/DeepCell2/prototypes/plots/072016_plots/BMDM_all.pdf'
plt.savefig(filename, format='pdf')
plt.close()





