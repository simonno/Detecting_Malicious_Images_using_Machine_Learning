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


if __name__ == "__main__":
    create_CSV(10000)
