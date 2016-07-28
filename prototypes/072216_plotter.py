import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl 
import seaborn as sns
mpl.rcParams['pdf.fonttype'] = 42

h1 = np.load('')['loss_history'][()]
h2 = np.load('')['loss_history'][()]
h3 = np.load('')['loss_history'][()]
h4 = np.load('')['loss_history'][()]
h5 = np.load('')['loss_history'][()]

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

epoch = np.arange(1, 26, 1)

sns.set_style("white")
solid = mlines.Line2D([], [], color='black', linestyle = '-', label = 'Training')
dashed = mlines.Line2D([],[], color='black', linestyle = '--', label= 'Validation')

plt.figure(0)
plt.plot(epoch, e1, 'k-', epoch, v1, 'k--', epoch, e2, 'k-', epoch, v2, 'k--', epoch, e3, 'k-', epoch, v3, 'k--', epoch, e4, 'k-', epoch, v4, 'k--', epoch, e5, 'k-', epoch, v5, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(handles=[solid,dashed])
plt.title('Training and validation error: bn_feature_net_81x81 on HeLa_all_81x81 set')
filename = '/home/nquach/DeepCell2/prototypes/plots/072216_plots/bn_feature_net_81x81.pdf'
plt.savefig(filename, format = 'pdf')
