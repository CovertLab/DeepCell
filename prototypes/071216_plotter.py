import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl 
import seaborn as sns
mpl.rcParams['pdf.fonttype'] = 42
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
'''
h21 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_feature_net_61x61_1.npz')['loss_history'][()]
h22 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_feature_net_61x61_2.npz')['loss_history'][()]
h23 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_feature_net_61x61_3.npz')['loss_history'][()]
'''
h31 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_dropout_1.npz')['loss_history'][()]
h32 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_dropout_2.npz')['loss_history'][()]
h33 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_dropout_3.npz')['loss_history'][()]
'''
h41 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_feature_net_61x61_dropout_1.npz')['loss_history'][()]
h42 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_feature_net_61x61_dropout_2.npz')['loss_history'][()]
h43 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_feature_net_61x61_dropout_3.npz')['loss_history'][()]
'''
h51 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-12_HeLa_all_61x61_bn_feature_net_61x61_1.npz')['loss_history'][()]
h52 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-12_HeLa_all_61x61_bn_feature_net_61x61_2.npz')['loss_history'][()]
h53 = np.load('/home/nquach/DeepCell2/trained_networks/HeLa/2016-07-12_HeLa_all_61x61_bn_feature_net_61x61_3.npz')['loss_history'][()]
'''
h61 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm1_1.npz')['loss_history'][()]
h62 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm2_2.npz')['loss_history'][()]
h63 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-11_HeLa_all_stdnorm_61x61_bn_stdnorm3_3.npz')['loss_history'][()]
'''
h71 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_fn_multires_61x61_1.npz')['loss_history'][()]
h72 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_fn_multires_61x61_2.npz')['loss_history'][()]
h73 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_fn_multires_61x61_3.npz')['loss_history'][()]
'''
h81 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_fn_multires_61x61_1.npz')['loss_history'][()]
h82 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_fn_multires_61x61_2.npz')['loss_history'][()]
h83 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_fn_multires_61x61_3.npz')['loss_history'][()]
'''
h91 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_shearing_1.npz')['loss_history'][()]
h92 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_shearing_2.npz')['loss_history'][()]
h93 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_feature_net_61x61_shearing_3.npz')['loss_history'][()]

h111 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_bn_multires_61x61_1.npz')['loss_history'][()]
h112 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_bn_multires_61x61_2.npz')['loss_history'][()]
h113 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_61x61_bn_multires_61x61_3.npz')['loss_history'][()]
'''
h101 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_feature_net_61x61_shearing_1.npz')['loss_history'][()]
h102 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_feature_net_61x61_shearing_2.npz')['loss_history'][()]
h103 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-12_HeLa_all_stdnorm_61x61_feature_net_61x61_shearing_3.npz')['loss_history'][()]



h121 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_1.npz')['loss_history'][()]
h122 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_2.npz')['loss_history'][()]
h123 = np.load('/home/nquach/DeepCell2/trained_networks/2016-07-09_HeLa_all_stdnorm_61x61_bn_multires_stdnorm_3.npz')['loss_history'][()]
'''
e11 = np.subtract(1, h11['acc'])
e12 = np.subtract(1, h12['acc'])
e13 = np.subtract(1, h13['acc'])
'''
e21 = np.subtract(1, h21['acc'])
e22 = np.subtract(1, h22['acc'])
e23 = np.subtract(1, h23['acc'])
'''
e31 = np.subtract(1, h31['acc'])
e32 = np.subtract(1, h32['acc'])
e33 = np.subtract(1, h33['acc'])
'''
e41 = np.subtract(1, h41['acc'])
e42 = np.subtract(1, h42['acc'])
e43 = np.subtract(1, h43['acc'])
'''
e51 = np.subtract(1, h51['acc'])
e52 = np.subtract(1, h52['acc'])
e53 = np.subtract(1, h53['acc'])
'''
e61 = np.subtract(1, h61['acc'])
e62 = np.subtract(1, h62['acc'])
e63 = np.subtract(1, h63['acc'])
'''
e71 = np.subtract(1, h71['acc'])
e72 = np.subtract(1, h72['acc'])
e73 = np.subtract(1, h73['acc'])
'''
e81 = np.subtract(1, h81['acc'])
e82 = np.subtract(1, h82['acc'])
e83 = np.subtract(1, h83['acc'])
'''
e91 = np.subtract(1, h91['acc'])
e92 = np.subtract(1, h92['acc'])
e93 = np.subtract(1, h93['acc'])

e111 = np.subtract(1, h111['acc'])
e112 = np.subtract(1, h112['acc'])
e113 = np.subtract(1, h113['acc'])
'''
e101 = np.subtract(1, h101['acc'])
e102 = np.subtract(1, h102['acc'])
e103 = np.subtract(1, h103['acc'])



e121 = np.subtract(1, h121['acc'])
e122 = np.subtract(1, h122['acc'])
e123 = np.subtract(1, h123['acc'])
'''
v11 = np.subtract(1, h11['val_acc'])
v12 = np.subtract(1, h12['val_acc'])
v13 = np.subtract(1, h13['val_acc'])
'''
v21 = np.subtract(1, h21['val_acc'])
v22 = np.subtract(1, h22['val_acc'])
v23 = np.subtract(1, h23['val_acc'])
'''
v31 = np.subtract(1, h31['val_acc'])
v32 = np.subtract(1, h32['val_acc'])
v33 = np.subtract(1, h33['val_acc'])
'''
v41 = np.subtract(1, h41['val_acc'])
v42 = np.subtract(1, h42['val_acc'])
v43 = np.subtract(1, h43['val_acc'])
'''
v51 = np.subtract(1, h51['val_acc'])
v52 = np.subtract(1, h52['val_acc'])
v53 = np.subtract(1, h53['val_acc'])
'''
v61 = np.subtract(1, h61['val_acc'])
v62 = np.subtract(1, h62['val_acc'])
v63 = np.subtract(1, h63['val_acc'])
'''
v71 = np.subtract(1, h71['val_acc'])
v72 = np.subtract(1, h72['val_acc'])
v73 = np.subtract(1, h73['val_acc'])
'''
v81 = np.subtract(1, h81['val_acc'])
v82 = np.subtract(1, h82['val_acc'])
v83 = np.subtract(1, h83['val_acc'])
'''
v91 = np.subtract(1, h91['val_acc'])
v92 = np.subtract(1, h92['val_acc'])
v93 = np.subtract(1, h93['val_acc'])

v111 = np.subtract(1, h111['val_acc'])
v112 = np.subtract(1, h112['val_acc'])
v113 = np.subtract(1, h113['val_acc'])
'''
v101 = np.subtract(1, h101['val_acc'])
v102 = np.subtract(1, h102['val_acc'])
v103 = np.subtract(1, h103['val_acc'])



v121 = np.subtract(1, h121['val_acc'])
v122 = np.subtract(1, h122['val_acc'])
v123 = np.subtract(1, h123['val_acc'])
'''
'''
##all vs stdnorm
# 1 vs 2 (vanilla)
data1 = [(e11, e12, e13), (v11, v12, v13), (e21, e22, e23), (v21, v22, v23)]
data1 = np.moveaxis(np.asarray(data1), 0, -1)

#3 vs 4 (dropout)
data2 = [(e31, e32, e33), (v31, v32, v33), (e41, e42, e43), (v41, v42, v43)]
data2 = np.moveaxis(np.asarray(data2), 0, -1)

#5 vs 6 (BN)
data3 = [(e51, e52, e53), (v51, v52, v53), (e61, e62, e63), (v61, v62, v63)]
data3 = np.moveaxis(np.asarray(data3), 0, -1)

#7 vs 8 (fn multires)
data4 = [(e71, e72, e73), (v71, v72, v73), (e81, e82, e83), (v81, v82, v83)]
data4 = np.moveaxis(np.asarray(data4), 0, -1)

#9 vs 10 (shearing)
data5 = [(e91, e92, e93), (v91, v92, v93), (e101, e102, e103), (v101, v102, v103)]
data5 = np.moveaxis(np.asarray(data5), 0, -1)

#11 vs 12 (bn multires)
data6 = [(e111, e112, e113), (v111, v112, v113), (e121, e122, e123), (v121, v122, v123)]
data6 = np.moveaxis(np.asarray(data6), 0, -1)
'''
## model vs model: all
#1 vs 3 (vanilla vs dropout) all
data7 = [(e11, e12, e13), (v11, v12, v13), (e31, e32, e33), (v31, v32, v33)]
data7 = np.moveaxis(np.asarray(data7), 0, -1)

#1 vs 5 (vanilla vs BN) all
data8 = [(e11, e12, e13), (v11, v12, v13), (e51, e52, e53), (v51, v52, v53)]
data8 = np.moveaxis(np.asarray(data8), 0, -1)

#1 vs 7 (vanilla vs fn multires) all
data9 = [(e11, e12, e13), (v11, v12, v13), (e71, e72, e73), (v71, v72, v73)]
data9 = np.moveaxis(np.asarray(data9), 0, -1)

#1 vs 9 (vanilla vs shearing) all
data10 = [(e11, e12, e13), (v11, v12, v13), (e91, e92, e93), (v91, v92, v93)]
data10 = np.moveaxis(np.asarray(data10), 0, -1)

#1 vs 11 (vanilla vs bn multires) all
data11 = [(e11, e12, e13), (v11, v12, v13), (e111, e112, e113), (v111, v112, v113)]
data11 = np.moveaxis(np.asarray(data11), 0, -1)
'''
## model vs model: stdnorm
#2 vs 4 (vanilla vs dropout) norm
data12 = [(e21, e22, e23), (v21, v22, v23), (e41, e42, e43), (v41, v42, v43)]
data12 = np.moveaxis(np.asarray(data12), 0, -1)

#2 vs 6 (vanilla vs BN) norm
data13 = [(e21, e22, e23), (v21, v22, v23), (e61, e62, e63), (v61, v62, v63)]
data13 = np.moveaxis(np.asarray(data13), 0, -1)

#2 vs 8 (vanilla vs fn multires) norm
data14 = [(e21, e22, e23), (v21, v22, v23), (e81, e82, e83), (v81, v82, v83)]
data14 = np.moveaxis(np.asarray(data14), 0, -1)

#2 vs 10 (vanilla vs shearing) norm
data15 = [(e21, e22, e23), (v21, v22, v23), (e101, e102, e103), (v101, v102, v103)]
data15 = np.moveaxis(np.asarray(data15), 0, -1)

#2 vs 12 (vanilla vs bn multires) norm
data16 = [(e21, e22, e23), (v21, v22, v23), (e121, e122, e123), (v121, v122, v123)]
data16 = np.moveaxis(np.asarray(data16), 0, -1)
'''
'''
print data1.shape
print data2.shape
print data3.shape
print data4.shape
print data5.shape
print data6.shape
print data7.shape
print data8.shape
print data9.shape
print data10.shape
print data11.shape
print data12.shape
print data13.shape
print data14.shape
print data15.shape
print data16.shape
'''

##plots of all vs stdnorm for each model

sns.set_style("white")
sns.set_palette("Paired")
ax_lim = [0, 25, 0.04, 0.22]
'''
plt.figure(0)
sns.tsplot(data=data1, condition=['no stdnorm: training', 'no stdnorm: validation','stdnorm: training', 'stdnorm: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: feature_net_61x61')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/feature_net_61x61.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(1)
sns.tsplot(data=data2, condition=['no stdnorm: training', 'no stdnorm: validation','stdnorm: training', 'stdnorm: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: feature_net_61x61 w/ Dropout')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/feature_net_61x61_dropout.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(2)
sns.tsplot(data=data3, condition=['no stdnorm: training', 'no stdnorm: validation','stdnorm: training', 'stdnorm: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: bn_feature_net_61x61')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/bn_feature_net_61x61.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(3)
sns.tsplot(data=data4, condition=['no stdnorm: training', 'no stdnorm: validation','stdnorm: training', 'stdnorm: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: feature_net_multires_61x61')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/feature_net_multires_61x61.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(4)
sns.tsplot(data=data5, condition=['no stdnorm: training', 'no stdnorm: validation','stdnorm: training', 'stdnorm: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: feature_net_61x61 w/ Shearing')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/feature_net_61x61_shearing.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(5)
sns.tsplot(data=data6, condition=['no stdnorm: training', 'no stdnorm: validation','stdnorm: training', 'stdnorm: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: bn_feature_net_multires_61x61')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/bn_feature_net_multires_61x61.pdf'
plt.savefig(filename, format='pdf')
plt.close()
'''
##plots of vanilla vs other, all

plt.figure(6)
sns.tsplot(data=data7, condition=['feature_net_61x61: training', 'feature_net_61x61: validation','feature_net_61x61 + Dropout: training', 'feature_net_61x61 + Dropout: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: Dropout')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/feature_net_61x61_dropout_all.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(7)
sns.tsplot(data=data8, condition=['feature_net_61x61: training', 'feature_net_61x61: validation','bn_feature_net_61x61: training', 'bn_feature_net_61x61: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: Batch Normalization')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/bn_feature_net_61x61_all.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(8)
sns.tsplot(data=data9, condition=['feature_net_61x61: training', 'feature_net_61x61: validation','feature_net_multires_61x61: training', 'feature_net_multires_61x61: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: Multiresolution net')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/feature_net_multires_61x61_all.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(9)
sns.tsplot(data=data10, condition=['feature_net_61x61: training', 'feature_net_61x61: validation','feature_net_61x61 + Shearing: training', 'feature_net_61x61 + Shearing: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: Shearing')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/feature_net_61x61_shear_all.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(10)
sns.tsplot(data=data11, condition=['feature_net_61x61: training', 'feature_net_61x61: validation','bn_feature_net_multires_61x61: training', 'bn_feature_net_multires_61x61: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: Batch Normalized Multires net')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/bn_feature_net_multires_61x61_all.pdf'
plt.savefig(filename, format='pdf')
plt.close()

##plots vanilla vs other, norm
'''
plt.figure(11)
sns.tsplot(data=data12, condition=['feature_net_61x61: training', 'feature_net_61x61: validation','feature_net_61x61 + Dropout: training', 'feature_net_61x61 + Dropout: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: Dropout & stdnorm')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/feature_net_61x61_dropout_norm.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(12)
sns.tsplot(data=data13, condition=['feature_net_61x61: training', 'feature_net_61x61: validation','bn_feature_net_61x61: training', 'bn_feature_net_61x61: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: Batch Normalization & stdnorm')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/bn_feature_net_61x61_norm.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(13)
sns.tsplot(data=data14, condition=['feature_net_61x61: training', 'feature_net_61x61: validation','feature_net_multires_61x61: training', 'feature_net_multires_61x61: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: Multires net & stdnorm')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/feature_net_multires_61x61_norm.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(14)
sns.tsplot(data=data15, condition=['feature_net_61x61: training', 'feature_net_61x61: validation','feature_net_61x61 + Shearing: training', 'feature_net_61x61 + Shearing: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: Shearing & stdnorm')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/feature_net_61x61_shear_norm.pdf'
plt.savefig(filename, format='pdf')
plt.close()

plt.figure(15)
sns.tsplot(data=data16, condition=['feature_net_61x61: training', 'feature_net_61x61: validation','bn_feature_net_multires_61x61: training', 'bn_feature_net_multires_61x61: validation'], err_style='ci_bars')
plt.xlabel('Epoch')
plt.ylabel('Average Model Error')
plt.title('Average Model Error: Batch Normalized Multires net & stdnorm')
plt.axis(ax_lim)
filename = '/home/nquach/DeepCell2/prototypes/plots/071216_plots/bn_feature_net_multires_61x61_norm.pdf'
plt.savefig(filename, format='pdf')
plt.close()
'''



