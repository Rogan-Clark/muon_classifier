'''a variety of functions for training and testing CNNs, designed to add noise to simultae real data when testing.'''


from __future__ import absolute_import, division, print_function

import matplotlib as mpl
import numpy as np
KERAS_BACKEND='tensorflow'
import keras
import os, tempfile, sys, glob, h5py
from keras.utils import HDF5Matrix
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Input, GaussianNoise, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation, Dropout
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import regularizers
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from  matplotlib.pyplot import cm
from sklearn.preprocessing import scale
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.metrics import binary_accuracy
from deepexplain.tensorflow import DeepExplain
#from ctapipe.image import tailcuts_clean
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import scipy.signal as signals
from itertools import cycle
from picturekiller import *
from random import seed
from random import randint
import ctapipe
from ctapipe.instrument import CameraGeometry
from ctapipe.core import Tool
from ctapipe.image import tailcuts_clean 
from ctapipe.image.geometry_converter import chec_to_2d_array, array_2d_to_chec
from astropy.io import fits
from traitlets import (Integer, Float, List, Dict, Unicode)
from traitlets import Int
import astropy.units as unit
import pickle
import pandas as pa


#gan_refiner._make_predict_function()
global trainevents
trainevents = []
global validevents
validevents = []
global testevents
testevents = []
global train2
train2 = []
global test2
test2 = []
global valid2
valid2 = []

def pos_to_index(pos, size):
        rnd = np.round((pos / size).to_value(unit.dimensionless_unscaled), 1)
        unique = np.sort(np.unique(rnd))
        mask = np.append(np.diff(unique) > 0.5, True)
        bins = np.append(unique[mask] - 0.5, unique[-1] + 0.5)
        return np.digitize(rnd, bins) - 1

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, cax=cax)

def make_mosaic(nimgs,imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    imshape = imgs.shape[:2]
    print(nimgs,imshape,nrows,ncols)
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[:,:,i]
    return mosaic

def get_confusion_matrix_one_hot(runname,model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    mr=[]
    mr2=[]
    mr3=[]
    print(model_results,truth)
    for x in model_results:
        mr.append(np.argmax(x))
        mr2.append(x)
    mr3 = label_binarize(mr, classes=[0, 1, 2])
    no_ev=min(len(mr),len(truth))
    print(no_ev)
    model_results=np.asarray(mr)[:no_ev]
    truth=np.asarray(truth)[:no_ev]
    print(np.shape(model_results),np.shape(truth))
    mr2=mr2[:no_ev]
    mr3=mr3[:no_ev]
    cm=confusion_matrix(y_target=truth,y_predicted=np.rint(np.squeeze(model_results)),binary=False)
    fig,ax=plot_confusion_matrix(conf_mat=cm,figsize=(5,5))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('/home/clarkr/Figures/'+runname+'confmat.png')
    lw=2
    n_classes=3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    t2 = label_binarize(truth, classes=[0, 1, 2])
    print(mr2[:100])
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(t2[:, i], np.asarray(mr2)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    fpr["micro"], tpr["micro"], _ = roc_curve(t2.ravel(), mr3.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.legend(loc="lower right")
    plt.savefig('/home/clarkr/Figures/'+runname+'_roc.png')
    np.save('/home/clarkr/confmatdata/'+runname+'_fp.npy',fpr)
    np.save('/home/clarkr/confmatdata/'+runname+'_tp.npy',tpr)
    return cm



def generate_training_sequences2d(onlyfiles, batch_size, batchflag, shilonflag=False):
    """ Generates training/test sequences on demand
    """
    nofiles = 0
    i = 0  # No. events loaded in total

    if batchflag == 'Train':
        filelist = onlyfiles[:652]
        global trainevents
        global train2
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            trainevents = trainevents + inputdata['event_id'][:].tolist()
            train2 = train2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Test':
        filelist = onlyfiles[652:1302]
        global testevents
        global test2
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            testevents = testevents + inputdata['event_id'][:].tolist()
            test2 = test2 + inputdata['id'][:].tolist()
            inputdata.close()
    else:
        print('Error: Invalid batchflag')
        raise KeyboardInterrupt

    while True:
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            chargearr = np.asarray(inputdata['squared_training'][:, 0, :, :])
#            chargearr = chargearr[:,8:40, 8:40]
            labelsarr = np.asarray(inputdata['event_label'][:])
            valilocs = np.where(labelsarr!=-1)[0]
            labelsarr = labelsarr[valilocs]
            chargearr = chargearr[valilocs,:,:]
            idarr = np.asarray(inputdata['id'][:])
            nofiles = nofiles + 1
            inputdata.close()
            chargearr = np.reshape(chargearr, (np.shape(chargearr)[0], 48, 48, 1))
            

            training_sample_count = len(chargearr)
            batches = int(training_sample_count / batch_size)
            remainder_samples = training_sample_count % batch_size
            i = i + 1000
            countarr = np.arange(0, len(labelsarr))
            ta2 = np.zeros((training_sample_count, 48, 48, 1))
            
            ta2[:, :, :, 0] = chargearr[:, :, :, 0]
            trainarr = ta2

            #displayarr=np.copy(trainarr)                #Create and plotdisplayarr to use in comparison plot later
            #plt.imshow(displayarr[0,:,:,0])
            #plt.show()

            choose=randint(0,2)  #Adds noise to the image
            if choose==0:
                trainarr=add_bad_flatfield(trainarr,training_sample_count)
            elif choose==1:
                trainarr=add_star(trainarr,training_sample_count)
            else:
                trainarr=add_noise(trainarr,5,training_sample_count)

            with open('camdata.pickle', 'rb') as handle:   #Requires cam data, stored as pickle file, to configure the image shape
                cam=pickle.load(handle) 
            for b in range(training_sample_count):         #Applies tailcut to image
                size = np.sqrt(cam.pix_area)
                col = pos_to_index(cam.pix_x, size)
                row = pos_to_index(cam.pix_y, size)
                #trainarrim=np.zeros(2304)
                #trainarrim=np.reshape(trainarrim,(48,48))
                trainarrim=trainarr[b,:,:,0]
                trainarrim=array_2d_to_chec(trainarrim)
                oof=trainarrim.shape                       #If statement here because some of the arrays reshape to 2047 long, not 2048. Extremely hacky move
                if oof[0]==2048:
                    cleanmask=tailcuts_clean(cam, trainarrim, picture_thresh=1.7, boundary_thresh=1.5, min_number_picture_neighbors=2) #Set same as in muon_writer_clean
                    trainarrim[~cleanmask]=0
                    #squared = np.full((row.max() + 1, col.max() + 1), np.nan)
                    #squared[row,col] = trainarrim
                    trainarrim=chec_to_2d_array(trainarrim)
                    trainarr[b,:,:,0]=trainarrim
                else: continue
            #plt.imshow(trainarr[0,:,:,0])
            #plt.show()
            '''difarray=trainarr-displayarr

            fig=plt.figure(figsize=(8,8))            #Plots a variety of comparative graphs to allow checking of noise addition. Leave commented out in proper test run
            columnsx=3
            rowsx=5
            for i in range(1,rowsx*columnsx+1,3):       
                fig.add_subplot(rowsx,columnsx,i)
                plt.imshow(displayarr[i,:,:,0])
                fig.add_subplot(rowsx,columnsx,i+1)
                plt.imshow(trainarr[i,:,:,0])
                fig.add_subplot(rowsx,columnsx,i+2)
                plt.imshow(difarray[i,:,:,0])     
            plt.show()
            exit()'''            
                

#            trainarr = (trainarr-np.amin(trainarr,axis=0))/(np.amax(trainarr,axis=0)-np.amin(trainarr,axis=0))    #Normalisation removed

            if remainder_samples:
                batches = batches + 1

            # generate batches of samples
            for idx in list(range(0, batches)):
                if idx == batches - 1:
                    batch_idxs = countarr[idx * batch_size:]
                else:
                    batch_idxs = countarr[idx *
                                          batch_size:idx *
                                          batch_size +
                                          batch_size]
                X = trainarr[batch_idxs]
                X = np.nan_to_num(X)
                Y = keras.utils.to_categorical(
                    labelsarr[batch_idxs], num_classes=3)
                yield (np.array(X), np.array(Y))
