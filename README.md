
# Detecting Malicious Images using Machine Learning
### The project's goal is to develop, through the Machine Learning tool, a solution to detect malicious image attacks.
The project consists two main stages:
1.	The first stage consists of learning Web weaknesses. We investigated the XSS and CSRF attacks. We developed vulnerable sites to the
    different types of weaknesses. Implements of an attack site and an illustration of the attack of XSS and CSRF attacks. 
2.	The second stage consists of the construction of stenography images, and the identification of stenography images using machine
    learning.
    
## Result achieved in the project
    Accuracy : 85.56 %
 
## Work Stages
### Building dataset of stenography images: 
1. Choosing CIFAR-10 dataset. 
2. Consists 50K 32x32 color images in 5 classes, with 10K images per class.
3. Select 25K images, and inject 384B malicious JS code to each image. 

### Writing the images to CSV file:
1. x.cvs and y.csv.
2. x.csv - each row contain the pixels' data in RGB format of each image.
3. y.csv - each row contain tag of each image, 1 - malicious image and 0 - clear image.

### Building The Machine Learning model:
1. Using XGBoost. 
2. Implement XGBoost model with depth of 256.
3. Features: pixels of the image.
4. Labels: 1 - malicious image and 0 - clear image


## Prerequisites
1.	Anaconda 3 (recomended)
    https://conda.io/docs/user-guide/install/download.html

2.	python 3 (anthor option)
    https://www.python.org/downloads/  

3. the following packages:
    1.  xgboost
        ```
        conda install -c conda-forge xgboost 
        ```
    2.	opencv-python
        ```
        conda install -c menpo opencv
        ```
    3.	sklearn
        ```
        conda install -c anaconda scikit-learn
        ```
    4.	cPickle
    5.	mxnet

4.  CIFAR-10 dataset (optinal)	
    The cifar-10 will be installed automatically on cloning.
    Download Manually From : https://www.cs.toronto.edu/~kriz/cifar.html


## Running The Project
1.  cloning the repository.
2.  run the "createData.py" file.
    create the images and convert them to csv files.
3.  run the "xgboost_model.py" file.
    run and train the model on the dataset.


## Authors

* **Noam Simon** - simonoam@gmail.com
* **Chen Eliyahou** - chen.eliyahou@gmail.com
* **Alon Shats** - alonshats49@gmail.com

