# Detecting_Malicious_Images_using_Machine_Learning
Detecting Malicious Images using Machine Learning
General description
The project's goal is to develop, through the Machine Learning tool, a solution to detect malicious image attacks.
The project consists two main stages:
  •	The first stage consists of learning Web weaknesses.  We investigated the XSS and CSRF attacks. We developed vulnerable sites to the different types of weaknesses. Implements of an attack site and an illustration of the attack of XSS and CSRF attacks. 
  •	The second stage consists of the construction of stenography images, and the identification of stenography images using machine learning.
    The levels to perform this task:
      o	Building dataset of stenography images. Choosing CIFAR-10 dataset. Consists 50K 32x32 color images in 5 classes, with 10K images per class.
        Select 25K images, and inject 384B malicious JS code to each image. Inject code every to the forth bit for each R, G, and B of the every RGB pixel.
      o	Using XGBoost – Machine Learning model, to detect malicious images.  First, we write the images to CSV file, and then we implement XGBoost model with depth of 256. We defined that features are the pixels. 

3.	Results achieved in the project:
    Accuracy : 85.56 %
    
4.	Instructions to install and run the program:
  a.	Installing – 
    a.	python 3
    b.	Anaconda 3
    c.	opencv-python
    d.	sklearn
    e.	cPickle
    f.	mxnet
    g.	torch
    h.	xgboost
  b.	Downloading – CIFR10 dataset. From : https://www.cs.toronto.edu/~kriz/cifar.html
  c.	Earlier information –
    a.	BadImagesFolder – the folder with the malicious codes that will inject in images.
    b.	cifar-10-batches-py – the folder that contains the data.
    c.	Image – the folder that contains all the images. Both malware and clear.
    d.	How to run the program:
    a.	Run "CreateMalware.py" file – In this running we create 50,000 images that 25,000 of them are malware. All the images creation in "image" folder.
    b.	Run "createCSV.py" file – In this running we write the images to CSV file, according to RGB model. Each image getting number between 0-255. The results are x.csv and y.csv files.
    c.	Run "xgboostMachineLearning.py" – this running doing the machine learning model by XGBoost, and prints the results on data test.


