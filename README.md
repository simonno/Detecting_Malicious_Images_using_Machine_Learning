# Detecting_Malicious_Images_using_Machine_Learning
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
  c.	Earlier information –
    a.	BadImagesFolder – the folder with the malicious codes that will inject in images.
    b.	cifar-10-batches-py – the folder that contains the data.
    c.	Image – the folder that contains all the images. Both malware and clear.
    d.	How to run the program:
    a.	Run "CreateMalware.py" file – In this running we create 50,000 images that 25,000 of them are malware. All the images creation in "image" folder.
    b.	Run "createCSV.py" file – In this running we write the images to CSV file, according to RGB model. Each image getting number between 0-255. The results are x.csv and y.csv files.
    c.	Run "xgboostMachineLearning.py" – this running doing the machine learning model by XGBoost, and prints the results on data test.

# Detecting Malicious Images using Machine Learning

The project's goal is to develop, through the Machine Learning tool, a solution to detect malicious image attacks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

    1.	python 3
    2.	Anaconda 3
    3.	opencv-python
    4.	sklearn
    5.	cPickle
    6.	mxnet
    7.	torch
    8.	xgboost
    9.	Downloading – CIFR10 dataset. From : https://www.cs.toronto.edu/~kriz/cifar.html

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

