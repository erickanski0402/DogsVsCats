from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
import numpy as np
import sqlite3 as lite
import populateDB as db
import glob
import errno
import random

conn = db.create_connection("data.db")

select_cats = """
                SELECT value_1, value_2, value_3
                FROM pixels
                WHERE id < 40001
              """
data = db.query(conn, select_cats)
print(np.asarray(data))

# TODO: since joins are being difficult. Current idea is to pull a list of
#   some number of random (but unique) examples of cats and the same for dogs.
#   Given that list of unique examples, pull the pixel values in one picture at
#   a time, filtered by the picture_id. Load them into a numpy array
