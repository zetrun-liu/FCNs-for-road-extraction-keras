# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 12:12:34 2019
带重叠的滑动窗口patch（256*256）预测并缝合形成大的mask（1024*1024）图片
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

fileDir = "data/membrane/IEEE_road/test/images"  #test images(1024*1024)
#fileDir = "data/membrane/train/f"
preDir = "data/membrane/IEEE_road/test/sub_test/predict/" #Dir of predict mask



#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def predict_z(src, predict_path):
    TEST_SET = os.listdir(src)
    model = D_resunet1()
    #model = res_unet1()
    print('Loading Model weights...')
    model.load_weights('D_resunet1.hdf5')
    print('completed!')
    img_h = 256
    img_w = 256
    stride = img_h-16
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        path1 = path[0:-7]+'mask.png'  #rename mask
        # load the image
        image = io.imread(os.path.join(src,path))
        h, w, _ = image.shape

        image = image / 255.0
        #image = img_to_array(image)
        # padding_img = (padding_img - np.min(padding_img)) / (np.max(padding_img) - np.min(padding_img))

        print('[{}/{}], predicting:{}'.format(n+1, len(TEST_SET), path))

        mask_whole = np.zeros((h, w, 1), dtype=np.uint8)
        #temp = np.zeros((img_h, img_h), dtype=np.uint8)

        for i in range(0, (h // stride)+1):
            for j in range(0, (w // stride)+1):
                h_begin = i * stride
                w_begin = j * stride
                
                if h_begin + img_h > h:
                    h_begin = h_begin - (h_begin + img_h - h)
                
                if w_begin + img_w > w:
                    w_begin = w_begin - (w_begin + img_w - w)
                
                crop = image[h_begin:h_begin + img_h, w_begin:w_begin + img_w, :3] #[****)
                
                ch, cw, _ = crop.shape

                if ch != img_h or cw != img_h:
                    print('invalid size!')
                    print(i, j, h_begin, w_begin, ch, cw)
                    break
                
                crop = np.expand_dims(crop, axis=0)
                pred = model.predict(crop, verbose=2)
                pred = pred.reshape((img_h, img_h, 1)).astype(np.float64)
                #pred = np.argmax(pred, axis=2)
                #print(pred.shape)
                #pred = np.array(pred)
                
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
                pred = pred * 255
                '''
                for a in range(img_h):
                    for b in range(img_h):
                        if pred[a, b] == 0.:
                            temp[a, b, :] = [223, 223, 223]
                        elif pred[a, b] == 1.:
                            temp[a, b, :] = [255, 204, 163]
                        else:
                            print('Unknown type:', pred[a, b])
                '''
                mask_whole[h_begin:h_begin + img_h, w_begin:w_begin + img_w] \
                    = pred
                # + mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size, :]
        cv2.imwrite(predict_path + path1, mask_whole[0:h, 0:w])
        #print('Done!')
        
predict_z(fileDir, preDir)        
