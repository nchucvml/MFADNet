#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import random as rn
import os,sys
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)


# =============================================================================
#  For reprodocable results, from keras.io
# =============================================================================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
tf.test.is_built_with_cuda()
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=5)
# from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
tf.keras.backend.set_session(sess)

import tensorflow.keras as keras
import glob
from tensorflow.keras.preprocessing import image as kImage
from sklearn.utils import compute_class_weight
from MFADNet import MFADNet
from tensorflow.keras.utils import get_file
#from tensorflow.keras.utils.data_utils import get_file
import gc
from datetime import datetime
from matplotlib import pyplot as plt
from scipy import sparse 
def getData(train_dir, dataset_dir):
    
    void_label = -1. # non-ROI
    
    d = ['train', 'test']
    X_list = []
    Y_list = []
    for i in range(len(d)):
        # given ground-truths, load inputs  
        y_list = glob.glob(os.path.join(train_dir, d[i], '*.png'))
        x_list = glob.glob(os.path.join(dataset_dir, d[i], '*.png'))

        if len(y_list)<=0 or len(x_list)<=0:
            print(os.path.join(dataset_dir, d[i]))
            print(os.path.join(train_dir, d[i]))
            raise ValueError('System cannot find the dataset path or ground-truth path. Please give the correct path.')
        if i == 0:
            flag = len(x_list)

        x_list_temp = []
        for j in range(len(y_list)):
            Y_name = os.path.basename(y_list[j])
            Y_name = Y_name.split('.')[0]
    #        Y_name = Y_name.split('gt')[1]
            for k in range(len(x_list)):
                X_name = os.path.basename(x_list[k])
                X_name = X_name.split('.')[0]
    #            X_name = X_name.split('in')[1]
                if (Y_name == X_name):
                    x_list_temp.append(x_list[k])
                    break
        x_list = x_list_temp

        if len(x_list)!=len(y_list):
            x_set = set(x_list)
            y_set = set(y_list)
            raise ValueError('The number of X_list and Y_list must be equal.', len(x_list), y_set.difference(x_set),len(y_list), x_set.difference(y_set))
            
        # X must be corresponded to Y
        x_list = sorted(x_list)
        y_list = sorted(y_list)

        X_list.extend(x_list)
        Y_list.extend(y_list)
    

    # load training data
    X = []
    #Y = []
    FY = []
    DY = []
    for i in range(len(X_list)):
        x = kImage.load_img(X_list[i])
        x = kImage.img_to_array(x)
        X.append(x)
        
        x = kImage.load_img(Y_list[i], grayscale = True)
        x = kImage.img_to_array(x)
        shape = x.shape
        x /= 255.0
        x = np.floor(x)
        FY.append(x)

        if x.all():
            DY.append(1)
        elif not x.any():
            DY.append(0)
        else:
            DY.append(2)
        
    X = np.asarray(X)
    FY = np.asarray(FY)
    DY = np.asarray(DY)

    idx1 = list(range(0, flag))
    np.random.shuffle(idx1)
    np.random.shuffle(idx1)
    idx2 = list(range(flag, X.shape[0]))
    np.random.shuffle(idx2)
    np.random.shuffle(idx2)
    idx = idx1+idx2

    X = X[idx]
    FY = FY[idx]
    DY = DY[idx]

    cls_weight_list = []
    for i in range(FY.shape[0]):
        y = FY[i].reshape(-1)
        idx = np.where(y!=void_label)[0]
        if(len(idx)>0):
            y = y[idx]
        lb = np.unique(y) #  0., 1
        cls_weight = compute_class_weight(class_weight='balanced', classes=lb , y=y)
        class_0 = cls_weight[0]
        class_1 = cls_weight[1] if len(lb)>1 else 1.0
        
        cls_weight_dict = {0:class_0, 1: class_1}
        cls_weight_list.append(cls_weight_dict)
    cls_weight_list = np.asarray(cls_weight_list)
    
    return [X, FY, DY, cls_weight_list, flag]

### training function    
def train(data, scene, mdl_path, vgg_weights_path):
    
    ### hyper-params
    lr = 1e-4
    # val_split = 0.2
    max_epoch = 10
    batch_size = 3
    loss_weights = [1, 1, 1]

    flag = data[4]
    X_train = data[0][:flag]
    X_test = data[0][flag:]
    FY_train = data[1][:flag]
    FY_test = data[1][flag:]
    DY_train = data[2][:flag]
    DY_test = data[2][flag:]
    ###

    img_shape = data[0][0].shape #(height, width, channel)
    fsn = MFADNet(lr, img_shape, scene, vgg_weights_path, loss_weights)
    fsn.initModel('CDnet')
    fsn.model.summary()
    fsn.mdl_compile()
    model = fsn.model
    
    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')
    # model.fit(X_train, {'frame_output':FY_train, 'class_output':DY_train, 'GAP': DY_train}, validation_data=(X_test, {'frame_output':FY_test, 'class_output':DY_test, 'GAP': DY_test}), epochs=max_epoch, batch_size=batch_size, 
    #           verbose=1, class_weight={'frame_output':data[3]})
    model.fit(X_train, {'frame_output':FY_train, 'class_output':DY_train, 'GAP': DY_train}, validation_data=(X_test, {'frame_output':FY_test, 'class_output':DY_test, 'GAP': DY_test}), epochs=max_epoch, batch_size=batch_size, 
             callbacks=[redu, early], verbose=1)
    #class_weight

    model.save(mdl_path)
    print(mdl_path)

    del model, data, early, redu


# =============================================================================
# Main func
# =============================================================================

dataset = {'CT': ['cbd']}

main_dir = os.path.join('./', 'MFADNet')
vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
if not os.path.exists(vgg_weights_path):
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg_weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP, cache_subdir='models',
                                file_hash='6d6bbae143d832006294945121d1f1fc')

main_mdl_dir = os.path.join(main_dir, 'models')
for category, scene_list in dataset.items():
    mdl_dir = os.path.join(main_mdl_dir, category)
    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)

    for scene in scene_list:
        print ('Training ->>> ' + category + ' / ' + scene)

        train_dir = './new_training_label'
        dataset_dir = './new_datasets'

        data = getData(train_dir, dataset_dir)
        
        mdl_path = os.path.join(mdl_dir, 'mdl_' + scene  + datetime.now().strftime('%Y%m%d%H%M%S') + '.h5')
        train(data, scene, mdl_path, vgg_weights_path)
        del data
        
    gc.collect()
