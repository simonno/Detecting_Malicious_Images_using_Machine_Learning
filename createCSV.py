import csv
import random
from PIL import Image
import numpy as np
import pandas as pd


def convert_array_to_csv(file_name, array):
    np_array = np.array(array)
    df = pd.DataFrame(np_array)
    df.to_csv(file_name + ".csv", header=None, index=None)


def create_CSV(number_of_images):
    images_array = []
    label_array = []
    malware_images_number = number_of_images / 2
    for i in range(number_of_images):
        image = Image.open("Images/image" + str(i) + ".png")
        image_array = np.array(image)
        new_image_array = []
        for row in image_array:
            for pixle in row:
                for column in pixle:
                    new_image_array.append(column)
        images_array.append(new_image_array)
        if i < malware_images_number:
            label_array.append([1])
        else:
            label_array.append([0])
        print(i)

    convert_array_to_csv("x", images_array)
    convert_array_to_csv("y", label_array)
    # np_array = np.array(images_array)
    # df = pd.DataFrame(np_array)
    # df.to_csv("classification_data.csv", header=None, index=None)

    # with open("classification_data.csv", "wb") as csv_file:
    #     writer = csv.writer(csv_file)
    #     newArray = []
    #     for i in range(number_of_images):
    #         image = Image.open("Images/image" + str(i) + ".png")
    #         image_array = np.array(image)
    #         output_string = ""
    #         for row in image_array:
    #             for column in row:
    #                 # try:
    #                 #     for value in column:
    #                 #         if value < 10:
    #                 #             value_string += "00" + str(value)
    #                 #         elif value < 100:
    #                 #             value_string += "0" + str(value)
    #                 #         else:
    #                 #             value_string += str(value)
    #                 # except:
    #                 output_string += str(column[0]) + ',' + str(column[1]) + ',' + str(column[2]) + ","
    #             output_string += ""
    #         if i < number_of_images:
    #             output_string += "1"
    #         else:
    #             output_string += "0"
    #         new_array_i = output_string.split(',')
    #         newArray.append(new_array_i)
    #         print(i)
    #     np_array = np.array(newArray)
    #     import pandas as pd
    #     df = pd.DataFrame(np_array)
    #     df.to_csv("classification_data.csv",  header=None)


if __name__ == "__main__":
    create_CSV(10000)
