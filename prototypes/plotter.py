import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''h11 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-06_HeLa_set1_61x61_nq_expt1_1.npz')['loss_history'][()]
h12 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-07_HeLa_set1_61x61_nq_expt1_2.npz')['loss_history'][()]
h13 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-07_HeLa_set1_61x61_nq_expt1_3.npz')['loss_history'][()]
h14 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-06_HeLa_set1_61x61_nq_expt1_4.npz')['loss_history'][()]
h15 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-06_HeLa_set1_61x61_nq_expt1_5.npz')['loss_history'][()]

h21 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-06_HeLa_set1_61x61_nq_expt2_1.npz')['loss_history'][()]
h22 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-06_HeLa_set1_61x61_nq_expt2_2.npz')['loss_history'][()]
h23 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-07_HeLa_set1_61x61_nq_expt2_3.npz')['loss_history'][()]
h24 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-07_HeLa_set1_61x61_nq_expt2_4.npz')['loss_history'][()]
h25 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-06_HeLa_set1_61x61_nq_expt2_5.npz')['loss_history'][()]

h31 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-06_HeLa_set1_61x61_nq_expt3_1.npz')['loss_history'][()]
h32 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-06_HeLa_set1_61x61_nq_expt3_2.npz')['loss_history'][()]
h33 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-08_HeLa_set1_61x61_nq_expt3_3.npz')['loss_history'][()]
h34 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-06_HeLa_set1_61x61_nq_expt3_4.npz')['loss_history'][()]
h35 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-07_HeLa_set1_61x61_nq_expt3_5.npz')['loss_history'][()]

h41 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-07_HeLa_set1_81x81_nq_expt4_1.npz')['loss_history'][()]
h42 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-07_HeLa_set1_81x81_nq_expt4_2.npz')['loss_history'][()]
h43 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-07_HeLa_set1_81x81_nq_expt4_3.npz')['loss_history'][()]
h44 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-06_HeLa_set1_81x81_nq_expt4_4.npz')['loss_history'][()]
h45 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-08_HeLa_set1_81x81_nq_expt4_5.npz')['loss_history'][()] '''

h1 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm1_1.npz')['loss_history'][()]
h2 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm2_2.npz')['loss_history'][()]
h3 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm3_3.npz')['loss_history'][()]
h4 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm4_4.npz')['loss_history'][()] 
h5 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm5_5.npz')['loss_history'][()]

mh1 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_1.npz')['loss_history'][()]
mh2 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_2.npz')['loss_history'][()]
mh3 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_3.npz')['loss_history'][()]
mh4 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_4.npz')['loss_history'][()]
mh5 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_5.npz')['loss_history'][()]

me1 = np.subtract(1, mh1['acc'])
me2 = np.subtract(1, mh2['acc'])
me3 = np.subtract(1, mh3['acc'])
me4 = np.subtract(1, mh4['acc'])
me5 = np.subtract(1, mh5['acc'])

mv1 = np.subtract(1, mh1['val_acc'])
mv2 = np.subtract(1, mh2['val_acc'])
mv3 = np.subtract(1, mh3['val_acc'])
mv4 = np.subtract(1, mh4['val_acc'])
mv5 = np.subtract(1, mh5['val_acc'])




'''e11 = np.subtract(1, h11['acc'])
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
v45 = np.subtract(1, h45['val_acc']) '''

e1 = np.subtract(1, h1['acc'])
e2 = np.subtract(1, h2['acc'])
e3 = np.subtract(1, h3['acc'])
e4 = np.subtract(1, h4['acc'])
e5 = np.subtract(1, h5['acc'])

v1 = np.subtract(1, h1['val_acc'])
v2 = np.subtract(1, h2['val_acc'])
v3 = np.subtract(1, h3['val_acc'])
v4 = np.subtract(1, h4['val_acc'])
v5 = np.subtract(1, h5['val_acc'])

#data = [(e11, e12, e13, e14, e15), (v11, v12, v13, v14, v15), (e21, e22, e23, e24, e25), (v21, v22, v23, v24, v25), (e31, e32, e33, e34, e35), (v31, v32, v33, v34, v35), (e41, e42, e43, e44, e45), (v41, v42, v43, v44, v45)]
data = [(e1, e2, e3, e4, e5), (v1, v2, v3, v4, v5), (me1, me2, me3, me4, me5), (mv1, mv2, mv3, mv4, mv5)]
data = np.asarray(data)
print data.shape
data = np.moveaxis(data, 0, -1)
print data.shape
sns.set_palette("Paired")
#sns.tsplot(data=data, condition=['feature_net_61x61 training', 'feature_net_61x61 validation', 'bn_feature_net_61x61 training', 'bn_feature_net_61x61 validation','bn_feature_net_61x61 w/ shearing, training', 'bn_feature_net_61x61 w/ shearing, validation','bn_feature_net_81x81 training', 'bn_feature_net_81x81 validation'], err_style='ci_bars')
sns.tsplot(data=data, condition=['bn_feature_net_61x61 training', 'bn_feature_net_61x61 validation','bn_feature_net_multires_61x61 training', 'bn_feature_net_multires_61x61 validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: HeLa_set_all_stdnorm_61x61 training set')

filename = '/home/nquach/DeepCell2/prototypes/plots/plotter_test.pdf'
plt.savefig(filename, format='pdf')

#, 'err21' : e21, 'err22' : e22, 'err23' : e23 , 'err24': e24, 'err25' : e25, 'err31' : e31, 'err32' : e32, 'err33' : e33, 'err34' : e34, 'err35' : e35, 'err41' : e41, 'err42' : e42, 'err43' : e43, 'err44' : e44, 'err45' : e45}



