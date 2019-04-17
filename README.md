#  FCNs-for-road-extraction-keras
Road extraction of high-resolution remote sensing images based on various semantic segmentation networks.

## Environment

**Win10 + Anaconda3 + tesndorflow-gpu + keras**

**Main packagess Required:** opencv-python, scikit-image, 

## Usage

**Here are the main steps of running the project:** 

Step1: Starting main.py to train the model and get the weights of model, which is a hdf5 type file;

Step2: Running sub_predict.py to predict the road of test data, of course you need to change a line of code for loading the corresponding model;

Step3: Using metrics.py to get the metrics of road segmentation performance. 
