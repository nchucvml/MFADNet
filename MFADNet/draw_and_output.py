# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:46:26 2022

@author: ya_han
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
#%%
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from instance_normalization import InstanceNormalization
from MFADNet import f_acc, d_acc
from matplotlib import pyplot
from tensorflow.keras.models import Model
def load_image(img_path, save_path):
    #x = image.load_img(path, grayscale=True)
    x = image.load_img(img_path)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)

    sp = os.path.join(save_path, os.path.basename(img_path))

    return x, sp

def load_image_path_list(path):
    walk = os.walk(path)
    imageList = []
    for path,dir_list,file_list in walk:
        for file_name in file_list:
            imageList.append(os.path.join(path, file_name))
    
    return imageList

def cam(model, x):
    prob = model.predict(x, 1, verbose=0)
    class_pred = float(prob[1])
    # channel attention
    last_conv_layer = model.get_layer('activation_6')
    # last_conv_layer = model.get_layer('activation_16')
    #conv_layer_output_value = last_conv_layer.output
    e = model.get_layer('activation_7').output
    # e = model.get_layer('activation_17').output
        
    # K.function() 讓我們可以藉由輸入影像至 `model.input` 得到 `last_conv_layer[0]` 的輸出值
    iterate = K.function([model.input], [e[0], last_conv_layer.output[0]])
    weight_value, conv_layer_output_value = iterate([x])

    # print(weight_value.shape)
    # print(conv_layer_output_value.shape)
    # 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
    for i in range(weight_value.shape[0]):
        conv_layer_output_value[:, :, i] *= (weight_value[i])
        
    # 計算 feature map 的 channel-wise 加總
    channel_heatmap = np.sum(conv_layer_output_value, axis=-1)
    
    # spatial attention
    h = model.get_layer('activation_8').output
    # h = model.get_layer('activation_18').output
    iterate = K.function([model.input], [h[0]])
    spatial_weight_value = iterate([x])
    spatial_heatmap = np.reshape(spatial_weight_value, channel_heatmap.shape)

    # result
    result_heatmap = np.multiply(channel_heatmap, spatial_heatmap)
    # return [channel_heatmap, spatial_heatmap, result_heatmap], class_pred

    return result_heatmap, class_pred

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def plot_bbox(heatmap, img_path, save_path, cla_pred):
    id = img_path.split('\\')[-1]
    #print(id)
    img = cv2.imread(img_path)
    prob_str = str(round(cla_pred, 4))

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap-np.min(heatmap)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255.0 * heatmap)

    # heatmap = np.uint8(cla_pred * heatmap)

    
    _, bImg = cv2.threshold(heatmap, 140, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # heatmap = np.uint8(cla_pred * heatmap)

    # contour_sizes = []
    # biggest_contour = 0
    # # print(np.round(cla_pred))
    if np.round(cla_pred)!=0:
        contour_sizes = [(cv2.contourArea(c), c) for c in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]


    dImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cImg = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    dImg_ori = cv2.addWeighted(dImg, 0.6, cImg, 0.4, 0.0)

    # previous_box = []
    pred_box = np.array([-1,-1,-1,-1])
    save_idx = ''
    text = []

    if np.round(cla_pred)!=0:
        if cv2.contourArea(biggest_contour) >= 30:
            x, y, w, h = cv2.boundingRect(biggest_contour)
            pred_box = np.array([x,y,x+w,y+h])
            text.append('('+str(x)+','+str(y)+','+str(x+w)+','+str(y+h)+')')
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    save_idx = 'CBD_box'+os.path.basename(save_path)
    save_idx_ori = 'CBD_heat'+os.path.basename(save_path)
    print(os.path.join(os.path.dirname(save_path), save_idx))
    cv2.imwrite(os.path.join(os.path.dirname(save_path), save_idx), img)
    cv2.imwrite(os.path.join(os.path.dirname(save_path), save_idx_ori), dImg_ori)




if __name__ == '__main__':
    #%%
    mdl_path = r"c:/Users/cvml_spare/Desktop/0907_CBD/MFADNet/MFADNet/models/CT/mdl_MFPM_cbd20230629154505.h5"
    CTFmdl = load_model(mdl_path, compile=False, custom_objects={'InstanceNormalization': InstanceNormalization, 'f_acc':f_acc, 'd_acc':d_acc}
                        ) #load the trained model
    #%%
    model_name = 'mdl_MFPM_cbd20220908095117'
    s_path = os.path.join('./MFADNet/test', model_name)
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    img_path_list = load_image_path_list('./new_datasets/test')
    for i in range(len(img_path_list)):
        print(i, img_path_list[i])
        img, save_path = load_image(img_path_list[i], s_path)
        heatmap, pred = cam(CTFmdl, img)
        plot_bbox(heatmap, img_path_list[i], save_path, pred)
