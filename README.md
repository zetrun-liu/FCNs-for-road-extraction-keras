# FCNs-for-road-extraction-keras
Road extraction of high-resolution remote sensing images based on various semantic segmentation networks
Here are the main steps of running the project:
Step1: Start train.py to get the weights of model, which is a hdf5 type file;
Step2: Running sub_predict.py to predict the road of test data, of course you need to load the corresponding model;
Step3: Using metrics.py to get the metrics of road segmentation performance. 
