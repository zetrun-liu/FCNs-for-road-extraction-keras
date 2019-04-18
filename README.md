#  FCNs-for-road-extraction-keras
Road extraction of high-resolution remote sensing images based on various semantic segmentation networks.

## Environment

**Win10 + Anaconda3 + tesndorflow-gpu + keras**

**Main packages Required:** opencv-python, scikit-image, 

## Details about the project

Due to FCNs can take arbitrary size image as input, however it will need amount of GPU memory to store the feature maps. Here, we utilize ﬁxed-sized training images (256×256) to train the model. These training images are sampled from the original images by sliding windows technique.

**data.py:** Used as a data generator;

**crop_image.py:** Got samples from the original images by sliding windows technique;

**model.py:** Contain various FCNs model, including **FCN-8s/2s, SegNet, Unet, VGGUnet, ResUnet and D-ResUnet**;

**metrics.py:** Calculating the metrics(precision/recall/active IoU) of the predicted road segmentation maps;

**sub_predict.py:**  In the original test images, sliding window technology with 16-pixels overlapping was used to predict each patch and splice them one by one to produce the final original size image segmentation image.


## Usage

**Here are the main steps of running the project:** 

Step1: Starting main.py to train the model and get the weights of model, which is a hdf5 type file;

Step2: Running sub_predict.py to predict the road of test data, of course you need to change a line of code for loading the corresponding model;

Step3: Using metrics.py to get the metrics of road segmentation performance. 

## Reference

1. https://github.com/HLearning/unet_keras;

2. https://github.com/zhixuhao/unet;

3. https://github.com/DavideA/dilation-keras;

4. https://github.com/mrgloom/awesome-semantic-segmentation
