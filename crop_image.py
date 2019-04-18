# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:07:33 2019
Note! crop the images and masks need to change several lines code(60/63)
@author: zetn
"""
from model import unet, segnet_vgg16, fcn_vgg16_8s, VGGUnet2, res_unet, res_unet1, D_resunet1
from data import trainGenerator, testGenerator, saveResult, testGenerator2
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import os, cv2
import numpy as np
import skimage.io as io
import skimage.transform as trans

fileDir = "data/membrane/test/sub_test/mask8"  #test images(1024*1024)
#fileDir = "data/membrane/train/f"
preDir = "data/membrane/test/masks_crops/" #Dir of predict mask

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def crop_image(src, save_path):
    TEST_SET = os.listdir(src)
    img_h = 256
    img_w = 256
    stride = img_h-40
    for n in range(len(TEST_SET)):
        image_name = TEST_SET[n]
        #path1 = image_name[0:-7]+'mask.png'  #rename mask
        # load the image
        #image = cv2.imread(os.path.join(src,image_name), cv2.IMREAD_UNCHANGED)
        image = cv2.imread(os.path.join(src,image_name))
        #image = io.imread(os.path.join(src,image_name))
        
        #print(image.shape)
        #h, w, _ = image.shape
        h, w = image.shape

        num = 0;
        #image = img_to_array(image)
        # padding_img = (padding_img - np.min(padding_img)) / (np.max(padding_img) - np.min(padding_img))

        print('[{}/{}], croping:{}'.format(n+1, len(TEST_SET), image_name))

        #mask_whole = np.zeros((h, w, 1), dtype=np.uint8)
        #temp = np.zeros((img_h, img_h), dtype=np.uint8)

        for i in range(0, (h // stride)+1):
            for j in range(0, (w // stride)+1):
                h_begin = i * stride
                w_begin = j * stride
                
                if h_begin + img_h > h:
                    h_begin = h_begin - (h_begin + img_h - h)
                
                if w_begin + img_w > w:
                    w_begin = w_begin - (w_begin + img_w - w)
                
                crop = image[h_begin:h_begin + img_h, w_begin:w_begin + img_w] 
                if num <= 9:
                    #path1 = image_name[0:-4]+'0'+ str(num)+'.jpg'
                    path1 = image_name[0:-4]+'0'+ str(num)+'.png'
                else:
                    #path1 = image_name[0:-4]+str(num)+'.jpg'
                    path1 = image_name[0:-4]+str(num)+'.png'
                
                #io.imsave(save_path + path1, crop)
                cv2.imwrite(save_path + path1, crop)
                num = num + 1
        #print('Done!')
crop_image(fileDir, preDir)               
