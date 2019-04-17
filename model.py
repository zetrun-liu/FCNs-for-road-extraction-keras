import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import keras.backend as K
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, concatenate, add, AtrousConvolution2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.losses import binary_crossentropy
from keras import backend as keras
from keras.layers import Dense, Flatten, ZeroPadding2D, BatchNormalization, Activation, Conv2DTranspose

file_path = os.path.dirname( os.path.abspath(__file__) )
VGG_Weights_path = file_path+"/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"

def IoU(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #y_true_f = np.array(K.flatten(y_true))
    #y_pred_f = np.array(K.flatten(y_pred))
    #y_pred_f[y_pred_f >= 0.5] = 1
    #y_pred_f[y_pred_f < 0.5] = 0
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

def final_loss(y_true, y_pred):
    loss1 = binary_crossentropy(y_true, y_pred)
    loss2 = 1 - IoU(y_true, y_pred)
    return loss1 + loss2

def segnet_vgg16(input_size = (256,256,3)):
    
    inputs = Input(input_size)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Up Block 1
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 2
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 3
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 4
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 5
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    
    model = Model(input = inputs, output = x)

    model.compile(optimizer = Adam(lr = 2e-4), loss = final_loss, metrics = [IoU])
    
    return model
    
def fcn_vgg16_8s(input_size = (256,256,3)):
    
    inputs = Input(input_size)
    x = BatchNormalization()(inputs)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    block_3 = Conv2D(1, (1, 1), activation='relu', padding='same')(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    block_4 = Conv2D(1, (1, 1), activation='relu', padding='same')(x)
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding="same")(x)

    block_5 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    
    sum_1 = add([block_4, block_5])
    sum_1 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(sum_1)
    
    sum_2 = add([block_3, sum_1])
    
    x = Conv2DTranspose(1, kernel_size=(16, 16), strides=(8, 8), activation='softmax', padding='same')(sum_2)
    
    model = Model(input = inputs, output = x)

    model.compile(optimizer = Adam(lr = 2e-4), loss = final_loss, metrics = [IoU])
    
    return model

def fcn_2s(input_size = (256,256,3)):
    
    inputs = Input(input_size)
    x = BatchNormalization()(inputs)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    block_1 = Conv2D(1, (1, 1), activation='relu', padding='same')(x)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    block_2 = Conv2D(1, (1, 1), activation='relu', padding='same')(x)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    block_3 = Conv2D(1, (1, 1), activation='relu', padding='same')(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    block_4 = Conv2D(1, (1, 1), activation='relu', padding='same')(x)
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding="same")(x)

    block_5 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    
    sum_1 = add([block_4, block_5])
    sum_1 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(sum_1)
    
    sum_2 = add([block_3, sum_1])
    sum_2 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(sum_2)
    
    sum_3 = add([block_2, sum_2])
    sum_3 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(sum_3) 
    
    sum_4 = add([block_1, sum_3])   
    x = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), activation='sigmoid', padding='same')(sum_4)
    
    model = Model(input = inputs, output = x)

    model.compile(optimizer = Adam(lr = 2e-4), loss = final_loss, metrics = [IoU])
    
    return model

def unet(pretrained_weights = None,input_size = (256,256,3)):
    
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 2e-4), loss = final_loss, metrics = [IoU])
    #model.compile(optimizer = SGD(lr=1e-3, decay=0.0, momentum=0.9, nesterov=True), loss = final_loss, metrics = [IoU])
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def VGGUnet2(input_size = (1024,1024,3)):
    
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(drop5) 
    
    conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    drop6 = Dropout(0.5)(conv6)
    
    up5 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop6))
    merge5 = concatenate([drop5,up5], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)    

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

def unet2(pretrained_weights = None,input_size = (256,256,3)):
    
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    #drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv5)
    #drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 2e-4), loss = final_loss, metrics = [IoU])
    #model.compile(optimizer = SGD(lr=1e-3, decay=0.0, momentum=0.9, nesterov=True), loss = final_loss, metrics = [IoU])
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path


def res_unet1(input_shape = (256,256,3)):
    inputs = Input(shape=input_shape)

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)
    
    model = Model(input=inputs, output=path)
    model.compile(optimizer = Adam(lr = 2e-4), loss = final_loss, metrics = [IoU])
    
    return model

def d_encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [512, 512], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder

def d_decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[3]], axis=3)
    main_path = res_block(main_path, [512, 512], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path

def d_res_unet1(input_shape = (256,256,3)):
    inputs = Input(shape=input_shape)

    to_decoder = d_encoder(inputs)

    path = res_block(to_decoder[3], [512, 512], [(2, 2), (1, 1)])

    path = d_decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)
    
    model = Model(input=inputs, output=path)
    model.compile(optimizer = Adam(lr = 2e-4), loss = final_loss, metrics = [IoU])
    
    return model

def D_resunet(input_size = (256,256,3)):

	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	inputs = Input(input_size)

	main_path = Conv2D(64, (3, 3), padding='same')(inputs)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
    
	main_path = Conv2D(64, (3, 3), padding='same')(main_path)
    
	shortcut = Conv2D(64, (1, 1))(inputs)
	shortcut = BatchNormalization()(shortcut)   
    
	main_path = add([shortcut, main_path])
   
	f0 = main_path
    
    #encoder res_block1
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(128, (3, 3), padding='same', strides=(2, 2))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(128, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(128, (1, 1), strides=(2, 2))(f0)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path]) 

	f1 = main_path        

    #encoder res_block2
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(256, (3, 3), padding='same', strides=(2, 2))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(256, (1, 1), strides=(2, 2))(f1)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path]) 

	f2 = main_path 
    
    #encoder res_block3
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(512, (3, 3), padding='same', strides=(2, 2))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(512, (1, 1), strides=(2, 2))(f2)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path]) 
	f3 = main_path
    
    #encoder res_block4
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(512, (3, 3), padding='same', strides=(2, 2))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(512, (1, 1), strides=(2, 2))(f3)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path]) 
	f4 = main_path
    
	'''
    #dilated_block
	dilate1 = AtrousConvolution2D(512, 3, 3, atrous_rate=(1, 1), activation='relu', border_mode='same')(main_path)
	#d1 = dilate1
	sum1 = add([f3, dilate1])
    
	dilate2 = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', border_mode='same')(dilate1)
	#d2 = dilate2
	sum2 = add([sum1, dilate2])
    
	dilate3 = AtrousConvolution2D(512, 3, 3, atrous_rate=(4, 4), activation='relu', border_mode='same')(dilate2)
	#d3 = dilate3
	sum3 = add([sum2, dilate3])
    
	dilate4 = AtrousConvolution2D(512, 3, 3, atrous_rate=(8, 8), activation='relu', border_mode='same')(dilate3)
	sum_dilate = add([sum3, dilate4])
    '''  
    
    #dilated_block
	dilate1 = Conv2D(512, (3, 3), dilation_rate=(1, 1), activation='relu', padding='same')(main_path)
	#d1 = dilate1
	sum1 = add([f4, dilate1])
    
	dilate2 = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same')(dilate1)
	#d2 = dilate2
	sum2 = add([sum1, dilate2])
    
	dilate3 = Conv2D(512, (3, 3), dilation_rate=(4, 4), activation='relu', padding='same')(dilate2)
	#d3 = dilate3
	sum3 = add([sum2, dilate3])
    
#	dilate4 = Conv2D(512, (3, 3), dilation_rate=(8, 8), activation='relu', padding='same')(dilate3)
#	sum_dilate = add([sum3, dilate4])

    #decoder part1
	main_path = UpSampling2D(size=(2, 2))(sum3)
	main_path = concatenate([main_path, f3], axis=3)
	o0 = main_path
    
    
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(512, (1, 1), strides=(1, 1))(o0)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path])    
    
    #decoder part2
	main_path = UpSampling2D(size=(2, 2))(main_path)
	main_path = concatenate([main_path, f2], axis=3)
	o1 = main_path
    
    
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(256, (1, 1), strides=(1, 1))(o1)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path])

    #decoder part3
	main_path = UpSampling2D(size=(2, 2))(main_path)
	main_path = concatenate([main_path, f1], axis=3)
	o2 = main_path
    
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(128, (3, 3), padding='same', strides=(1, 1))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(128, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(128, (1, 1), strides=(1, 1))(o2)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path])
    
    #decoder part4
	main_path = UpSampling2D(size=(2, 2))(main_path)
	main_path = concatenate([main_path, f0], axis=3)
	o3 = main_path
    
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(64, (3, 3), padding='same', strides=(1, 1))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(64, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(64, (1, 1), strides=(1, 1))(o3)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path])
    
	main_path = Conv2D(1, (1, 1), activation='sigmoid')(main_path)
    
	model = Model(input=inputs, output=main_path)
	model.compile(optimizer = Adam(lr = 2e-4), loss = final_loss, metrics = [IoU])
    
	return model

def D_resunet1(input_size = (256,256,3)):

	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	inputs = Input(input_size)

	main_path = Conv2D(64, (3, 3), padding='same')(inputs)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
    
	main_path = Conv2D(64, (3, 3), padding='same')(main_path)
    
	shortcut = Conv2D(64, (1, 1))(inputs)
	shortcut = BatchNormalization()(shortcut)   
    
	main_path = add([shortcut, main_path])
   
	f0 = main_path
    
    #encoder res_block1
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(128, (3, 3), padding='same', strides=(2, 2))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(128, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(128, (1, 1), strides=(2, 2))(f0)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path]) 

	f1 = main_path        

    #encoder res_block2
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(256, (3, 3), padding='same', strides=(2, 2))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(256, (1, 1), strides=(2, 2))(f1)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path]) 

	f2 = main_path 
    
    #encoder res_block3
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(512, (3, 3), padding='same', strides=(2, 2))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(512, (1, 1), strides=(2, 2))(f2)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path]) 
	f3 = main_path
    
	'''
    #dilated_block
	dilate1 = AtrousConvolution2D(512, 3, 3, atrous_rate=(1, 1), activation='relu', border_mode='same')(main_path)
	#d1 = dilate1
	sum1 = add([f3, dilate1])
    
	dilate2 = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', border_mode='same')(dilate1)
	#d2 = dilate2
	sum2 = add([sum1, dilate2])
    
	dilate3 = AtrousConvolution2D(512, 3, 3, atrous_rate=(4, 4), activation='relu', border_mode='same')(dilate2)
	#d3 = dilate3
	sum3 = add([sum2, dilate3])
    
	dilate4 = AtrousConvolution2D(512, 3, 3, atrous_rate=(8, 8), activation='relu', border_mode='same')(dilate3)
	sum_dilate = add([sum3, dilate4])
    '''  
    
    #dilated_block
	dilate1 = Conv2D(512, (3, 3), dilation_rate=(1, 1), activation='relu', padding='same')(main_path)
	#d1 = dilate1
	sum1 = add([f3, dilate1])
    
	dilate2 = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same')(dilate1)
	#d2 = dilate2
	sum2 = add([sum1, dilate2])
    
	dilate3 = Conv2D(512, (3, 3), dilation_rate=(4, 4), activation='relu', padding='same')(dilate2)
	#d3 = dilate3
	sum3 = add([sum2, dilate3])
    
	dilate4 = Conv2D(512, (3, 3), dilation_rate=(8, 8), activation='relu', padding='same')(dilate3)
	sum_dilate = add([sum3, dilate4])
    
    
    #decoder part1
	main_path = UpSampling2D(size=(2, 2))(sum_dilate)
	main_path = concatenate([main_path, f2], axis=3)
	o1 = main_path
    
    
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(256, (1, 1), strides=(1, 1))(o1)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path])

    #decoder part2
	main_path = UpSampling2D(size=(2, 2))(main_path)
	main_path = concatenate([main_path, f1], axis=3)
	o2 = main_path
    
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(128, (3, 3), padding='same', strides=(1, 1))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(128, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(128, (1, 1), strides=(1, 1))(o2)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path])
    
    #decoder part3
	main_path = UpSampling2D(size=(2, 2))(main_path)
	main_path = concatenate([main_path, f0], axis=3)
	o3 = main_path
    
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(64, (3, 3), padding='same', strides=(1, 1))(main_path)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)
	main_path = Conv2D(64, (3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(64, (1, 1), strides=(1, 1))(o3)
	shortcut = BatchNormalization()(shortcut) 
    
	main_path = add([shortcut, main_path])
    
	main_path = Conv2D(1, (1, 1), activation='sigmoid')(main_path)
    
	model = Model(input=inputs, output=main_path)
	model.compile(optimizer = Adam(lr = 2e-4), loss = final_loss, metrics = [IoU])
    
	return model

