import pickle as cPickle
import os
import mxnet as mx
import numpy as np
import cv2


def extract_images_and_labels(path, file):
    f = open(path + file, 'rb')
    dict = cPickle.load(f, encoding='latin1')
    images = dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    image_array = mx.nd.array(images)
    label_array = mx.nd.array(labels)
    return image_array, label_array


def extract_categories(path, file):
    f = open(path + file, 'rb')
    dict = cPickle.load(f, encoding='latin1')
    return dict['label_names']


def save_cifar_image(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1, 2, 0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path + file + ".png", array)


def create_images(number_of_images):
    images_array, labels_array = extract_images_and_labels("cifar/", "data_batch_1")
    # print(images_array.shape)
    # print(labels_array.shape)

    categories = extract_categories("cifar/", "batches.meta")
    path = os.path.join(os.path.expanduser('~'), 'PycharmProjects/FinalProject/Images')
    cats = []
    for i in range(number_of_images):
        save_cifar_image(images_array[i], path, "/image" + str(i))
        category = labels_array[i].asnumpy()
        category = int(category[0])
        cats.append(categories[category])
    return cats
