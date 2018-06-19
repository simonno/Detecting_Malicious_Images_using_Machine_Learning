
# Detecting Malicious Images using Machine Learning
### The project's goal is to develop, through the Machine Learning tool, a solution to detect malicious image attacks.
The project consists two main stages:
1.	The first stage consists of learning Web weaknesses. We investigated the XSS and CSRF attacks. We developed vulnerable sites to the
    different types of weaknesses. Implements of an attack site and an illustration of the attack of XSS and CSRF attacks. 
2.	The second stage consists of the construction of stenography images, and the identification of stenography images using machine
    learning.
    
## Results achieved in the project:
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

