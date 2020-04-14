#Script for loading a model and testing it with new data, currently built to add noise to data to simulate real events.
#Works with net_utils_48_noise, which requires picturekiller
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
mpl.use('TkAgg')                      #Depending on the day of the week, this may need to be 'Agg'
import numpy as np
import h5py
import keras
import os, tempfile, sys
from keras.utils import HDF5Matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Input, GaussianNoise
from keras.models import Model
from keras.layers import concatenate
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import regularizers
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from  matplotlib.pyplot import cm
from sklearn.preprocessing import scale
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from  matplotlib.pyplot import cm
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.metrics import binary_accuracy
from deepexplain.tensorflow import DeepExplain
from deeputils import plot
from net_utils_48_noise import *
#from ctapipe.image import tailcuts_clean
import time

runname='traincleantestraw'

plt.ioff()
#os.environ['CUDA_VISIBLE_DEVICES'] = '' # Uncomment this to use cpu rather than gpu.

global onlyfiles
onlyfiles=sorted(glob.glob('/store/rogansims/*.hdf5'))
onlyfiles=onlyfiles[:-1]
print(onlyfiles)
lay1=0
global Trutharr
Trutharr=[]
Trainarr=[]

#This defines the training/test sample based on the number of files. Make sure you set the same values in net_utils.py
for file in onlyfiles[:652]:
    inputdata=h5py.File(file,'r')
    labelsarr=np.asarray(inputdata['event_label'][:])
    labelsarr=labelsarr[np.where(labelsarr!=-1)]
    for value in labelsarr:
        Trainarr.append(value)
    inputdata.close()

for file in onlyfiles[652:1303]:
    inputdata=h5py.File(file,'r')
    labelsarr=np.asarray(inputdata['event_label'][:])
    labelsarr=labelsarr[np.where(labelsarr!=-1)]
    for value in labelsarr:
        Trutharr.append(value)
    inputdata.close()


opt=keras.optimizers.Adadelta()

model=load_model('/home/clarkr/Models/2dgantrain_muon100_clean.hdf5')    #Specify 2dgantrain model from elsewhere
print(model.summary())


print('Predicting')
starttime=time.time()
pred=model.predict_generator(generate_training_sequences2d(onlyfiles,50,'Test'),verbose=1,use_multiprocessing=False,steps=len(Trutharr)/50.0)
np.save('/home/clarkr/predictions/pred_'+runname+'.npy',pred)
endtime=time.time()
print('Evaluating')
score = model.evaluate_generator(generate_training_sequences2d(onlyfiles,50,'Test'),use_multiprocessing=False,steps=len(Trutharr)/50.0, verbose=1)
#model.save('/home/clarkr/Models/2dgantrain_'+runname+'.hdf5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

with DeepExplain(session=K.get_session()) as de:
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
    target_tensor = fModel(input_tensor)
    itgen=generate_training_sequences2d(onlyfiles,128,'Test')
    print(itgen)
    for xgen,ygen in itgen:
        xs=xgen
        ys=ygen
        break
    y2=np.zeros((1,128))
    for i in np.arange(len(ys)):
        y2[0,i]=int(np.argmax(ys[i]))
        
    ys=y2
    print(ys)
    print(np.shape(xs),np.shape(ys))
    attributions = de.explain('elrp', target_tensor * ys, input_tensor, xs)

from deeputils import plot, plt, newplot
x2=np.asarray(xs[:20])
xs=np.asarray(xs[:20],dtype=np.uint16)
ys=ys[:20]
attributions=attributions[:20]
n_cols = 4
#n_rows = 6
n_rows = int(len(attributions) / 2)


fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3*n_cols, 3*n_rows))

for i, a in enumerate(attributions):
    row, col = divmod(i, 2)
    print(xs[i])
    print('x2',x2[i])
    newplot(x2[i].reshape(48, 48), xi=None, axis=axes[row,col*2]).set_title('Original')
    plot(a.reshape(48,48), xi=None, axis=axes[row,col*2+1]).set_title('Attributions, label='+str(int(ys[0,i])))

plt.savefig('/home/clarkr/Figures/muontag_'+runname+'.png')

print(get_confusion_matrix_one_hot(runname,pred,Trutharr,))
print('Network Complete')
elaptime=endtime-starttime
print(elaptime)               #Prints prediction time
