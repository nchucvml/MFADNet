#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 2018

@author: longang
"""

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Activation, SpatialDropout2D, Dense, Flatten, Reshape, BatchNormalization
from tensorflow.keras.layers import Conv2D, Cropping2D, UpSampling2D
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import concatenate, add, multiply, Lambda, Cropping2D
from tensorflow.keras.regularizers import l2
from instance_normalization import InstanceNormalization
import tensorflow.keras.backend as K
import tensorflow as tf

def d_acc(dy_true, dy_pred):
    return K.equal(dy_true, K.round(dy_pred))

# def f_loss(fy_true, fy_pred):
#     return K.mean(K.binary_crossentropy(fy_true, fy_pred), axis=-1)

def f_acc(fy_true, fy_pred):
    return K.mean(K.equal(fy_true, K.round(fy_pred)), axis=-1)

class MFADNet(object):
    
    def __init__(self, lr, img_shape, scene, vgg_weights_path, loss_weights):
        self.lr = lr
        self.img_shape = img_shape
        self.scene = scene
        self.vgg_weights_path = vgg_weights_path
        self.loss_weights = loss_weights
        self.method_name = 'MFADNet'
        
    def VGG16(self, x): 
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        a = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        b = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Dropout(0.5, name='dr1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Dropout(0.5, name='dr2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Dropout(0.5, name='dr3')(x)
        
        return x, a, b
    
    def decoder(self,x,a,b):
        a = GlobalAveragePooling2D()(a)
        b = Conv2D(64, (1, 1), strides=1, padding='same')(b)
        b = GlobalAveragePooling2D()(b)
        
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x1 = multiply([x, b])
        x = add([x, x1])
        x = UpSampling2D(size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x2 = multiply([x, a])
        x = add([x, x2])
        x = UpSampling2D(size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        
        # Channel attetion
        c = GlobalAveragePooling2D()(x)
        d = GlobalMaxPooling2D()(x)
        shared_layer_one = Dense(32, activation='relu')
        shared_layer_two = Dense(64, activation='relu')
        c = shared_layer_two(shared_layer_one(c))
        d = shared_layer_two(shared_layer_one(d))
        e = add([c, d])
        e = Activation('sigmoid')(e)
        x = multiply([x, e])
        
        # Spatial Attention
        f = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(x)
        g = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(x)
        h = concatenate([f, g], axis=-1)
        h = Conv2D(1, (7,7), strides=1, padding='same')(h)
        h = Activation('sigmoid')(h)
        y = multiply([x, h])
        
        x = Conv2D(1, 1, padding='same', activation='sigmoid', name='frame_output')(y)
        return x, h
    
    def M_FPM(self, x):
        pool = MaxPooling2D((2, 2), strides=(1,1), padding='same')(x)
        pool = Conv2D(64, (1, 1), padding='same')(pool)
        
        d1 = Conv2D(64, (3, 3), padding='same')(x)
        
        y = concatenate([x, d1], axis=-1, name='cat4')
        y = Activation('relu')(y)
        d4 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(y)
        
        y = concatenate([x, d4], axis=-1, name='cat8')
        y = Activation('relu')(y)
        d8 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y)
        
        y = concatenate([x, d8], axis=-1, name='cat16')
        y = Activation('relu')(y)
        d16 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y)
        
        x = concatenate([pool, d1, d4, d8, d16], axis=-1)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.25)(x)
        return x
    
    def classifier(self, x):

        x = Flatten()(x)
        x = Dense(256,activation='relu',kernel_regularizer=l2(0.0002))(x)
        x = Dense(256,activation='relu',kernel_regularizer=l2(0.0002))(x)
        x = Dropout(0.7)(x)
        x = Dense(1, activation='softmax', kernel_regularizer=l2(0.0002), name='class_output')(x)
        return x
    
    def initModel(self, dataset_name):
        assert len(self.img_shape)==3
        h, w, d = self.img_shape
        
        net_input = Input(shape=(h, w, d), name='net_input')
        vgg_output = self.VGG16(net_input)
        vgg_model = Model(inputs=net_input, outputs=vgg_output, name='model')
        vgg_model.load_weights(self.vgg_weights_path, by_name=True)
        
        unfreeze_layers = ['block4_conv1','block4_conv2', 'block4_conv3']
        for layer in vgg_model.layers:
            if(layer.name not in unfreeze_layers):
                layer.trainable = False
            
        x,a,b = vgg_model.output

        x = self.M_FPM(x)
        x, self.spa = self.decoder(x,a,b)
        z = GlobalAveragePooling2D(name='GAP')(x)
        y = self.classifier(x)
        self.model = Model(inputs=net_input, outputs=[x, y, z], name='vision_model')
        print('finished init model')
    
    def f_loss(self, fy_true, fy_pred):
        fl = K.mean((1-fy_true)*fy_pred)
        l5 = K.mean((1-self.spa)*fy_pred)
        return 0.5*fl + l5
    
    def d_loss(self, dy_true, dy_pred):
        dl = K.binary_crossentropy(dy_true, dy_pred)
        return dl

    
    def mdl_compile(self):
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.)
        
        # loss_func = self.model_loss()
        c_loss = {'frame_output': self.f_loss,
                  'class_output': self.d_loss,
                  'GAP': self.d_loss}
        loss_weights = {'frame_output': self.loss_weights[0],
                        'class_output': self.loss_weights[1],
                        'GAP': self.loss_weights[2]}
        c_acc = {'frame_output': f_acc,
                  'class_output': d_acc,
                  'GAP': d_acc}

        self.model.compile(loss=c_loss, optimizer=opt, metrics=c_acc, loss_weights=loss_weights)
        print('finished compile')
