from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import glob
import errno
import random
import itertools

def convert(files):
    data_counts = [0, 0]
    count = 0

    for i in range(25000):
        if count % 2500 == 0:
            print("Opening new file")
            file = open("../Data/DogsVsCats/grayscale_csv_sets_10/Set_" + str(int(count / 1000)) + ".csv", "w")
        count += 1

        if(data_counts[0] > data_counts[1]):
            name = files + "dog" + str(data_counts[1]) + ".jpg"
            data_counts[1] += 1
        else:
            name = files + "cat" + str(data_counts[0]) + ".jpg"
            data_counts[0] += 1

        try:
            img = load_img(name, color_mode = "grayscale", target_size = (200,200))
            data = img_to_array(img).tolist()

            example = []
            for row in data:
                for col in row:
                    for value in col:
                        example.append(str(value))

            if "dog" in name:
                example.insert(0, "1")
            else:
                example.insert(0, "0")

            file.write(",".join(example) + "\n")
        except IOError as e:
            if e.errno != errno.EISDIR:
                print("error")


convert("../Data/DogsVsCats/train/")
