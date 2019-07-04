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
        if count % 500 == 0:
            file = open("../Data/DogsVsCats/csv_sets/Set_" + str(count) + ".csv", "w")
            print("Opening new file")
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
