#Loads npy data and plots ROC curves
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle

n_classes=3
lw=2
fpr1=np.load('/home/clarkr/confmatdata/muon_100_48_train_fp.npy', allow_pickle=True)
tpr1=np.load('/home/clarkr/confmatdata/muon_100_48_train_tp.npy', allow_pickle=True)
fpr2=np.load('/home/clarkr/confmatdata/raw_picturekiller_fp.npy', allow_pickle=True)
tpr2=np.load('/home/clarkr/confmatdata/raw_picturekiller_tp.npy', allow_pickle=True)
fpr3=np.load('/home/clarkr/confmatdata/muon100_clean_train_fp.npy', allow_pickle=True)
tpr3=np.load('/home/clarkr/confmatdata/muon100_clean_train_tp.npy', allow_pickle=True)
fpr4=np.load('/home/clarkr/confmatdata/testcleannoise_fp.npy', allow_pickle=True)
tpr4=np.load('/home/clarkr/confmatdata/testcleannoise_tp.npy', allow_pickle=True)



fpr1=fpr1.item()
tpr1=tpr1.item()
fpr2=fpr2.item()
tpr2=tpr2.item()
fpr3=fpr3.item()
tpr3=tpr3.item()
fpr4=fpr4.item()
tpr4=tpr4.item()
roc_auc1= dict()
roc_auc2=dict()
roc_auc3=dict()
roc_auc4=dict()

for i in range(n_classes):
    roc_auc1[i] = auc(fpr1[i], tpr1[i])
roc_auc1["macro"] = auc(fpr1["macro"], tpr1["macro"])
roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])

for i in range(n_classes):
    roc_auc2[i] = auc(fpr2[i], tpr2[i])
roc_auc2["macro"] = auc(fpr2["macro"], tpr2["macro"])
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])

for i in range(n_classes):
    roc_auc3[i] = auc(fpr3[i], tpr3[i])
roc_auc3["macro"] = auc(fpr3["macro"], tpr3["macro"])
roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])

for i in range(n_classes):
    roc_auc4[i] = auc(fpr4[i], tpr4[i])
roc_auc4["macro"] = auc(fpr4["macro"], tpr4["macro"])
roc_auc4["micro"] = auc(fpr4["micro"], tpr4["micro"])


plt.plot(fpr1["micro"], tpr1["micro"],
             label='Micro-average ROC curve of the raw data (area = {0:0.2f})'
                   ''.format(roc_auc1["micro"]),
             color='dodgerblue', linestyle='solid', linewidth=2)

plt.plot(fpr2["micro"], tpr2["micro"],
             label='Micro-average ROC curve of the uncleaned noisy data (area = {0:0.2f})'
                   ''.format(roc_auc2["micro"]),
             color='mediumorchid', linestyle='solid', linewidth=2)

plt.plot(fpr3["micro"], tpr3["micro"],
             label='Micro-average ROC curve of the cleaned raw data(area = {0:0.2f})'
                   ''.format(roc_auc3["micro"]),
             color='coral', linestyle='solid', linewidth=2)

plt.plot(fpr4["micro"], tpr4["micro"],
             label='Micro-average ROC curve of the cleaned noisy data (area = {0:0.2f})'
                   ''.format(roc_auc4["micro"]),
             color='seagreen', linestyle='solid', linewidth=2)

plt.plot(fpr1["macro"], tpr1["macro"],label='Macro-average ROC curve of the raw data (area = {0:0.2f})'
                   ''.format(roc_auc1["macro"]),
             color='dodgerblue', linestyle=':', linewidth=2)


plt.plot(fpr2["macro"], tpr2["macro"],label='Macro-average ROC curve of the noisy data (area = {0:0.2f})'
                   ''.format(roc_auc2["macro"]),
             color='mediumorchid', linestyle=':', linewidth=2)

plt.plot(fpr3["macro"], tpr3["macro"],label='Macro-average ROC curve of the cleaned raw data  (area = {0:0.2f})'
                   ''.format(roc_auc3["macro"]),
             color='coral', linestyle=':', linewidth=2)


plt.plot(fpr4["macro"], tpr4["macro"],label='Macro-average ROC curve of the cleaned noisy data (area = {0:0.2f})'
                   ''.format(roc_auc4["macro"]),
             color='seagreen', linestyle=':', linewidth=2)

#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#for i, color in zip(range(n_classes), colors):
#    plt.plot(fpr_un[i], tpr_un[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc_un[i]))

plt.legend(loc="lower right")
plt.show()
