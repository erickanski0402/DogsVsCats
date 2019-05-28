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
import glob
import errno
import random

def load_data(files):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    examples = []
    labels = []
    data_counts = [0, 0]
    count = 0

    print('Collecting pictures')
    for name in files:
        try:
            img = load_img(name, color_mode = "grayscale", target_size = (200,200))
            data = img_to_array(img)
            examples.append(data.tolist())

            if "dog" in name:
                labels.append(0)
            else:
                labels.append(0.99)
        except IOError as e:
            if e.errno != errno.EISDIR:
                print("error")

        count += 1
        if count % 50 == 0:
            print(count)

    print('Preparing data')
    balanced_examples_num = len(examples) * 0.8 / 2.0

    while examples:
        rand = random.randint(0, len(examples) - 1)

        # Need to figure out how to balance the training set
        #       ie equal number of cats and dogs

        if labels[rand] > 0 and data_counts[0] < balanced_examples_num:
            X_train.append(examples[rand])
            y_train.append(labels[rand])

            del examples[rand]
            del labels[rand]

            data_counts[0] += 1
        elif labels[rand] == 0 and data_counts[1] < balanced_examples_num:
            X_train.append(examples[rand])
            y_train.append(labels[rand])

            del examples[rand]
            del labels[rand]

            data_counts[1] += 1
        elif data_counts[0] == balanced_examples_num and data_counts[1] == balanced_examples_num:
            X_test.append(examples[rand])
            y_test.append(labels[rand])

            del examples[rand]
            del labels[rand]
    return [X_train, X_test, y_train, y_test]

data = load_data(glob.glob("./Data/pets_subset/*.jpg"))

X_train = np.subtract(np.asarray(data[0]), 127.0)
X_test = np.subtract(np.asarray(data[1]), 127.0)
y_train = np.asarray(data[2])
y_test = np.asarray(data[3])

# Use keras to preprocess each image and feed through convolutions
# and dense nn layers
print('Creating Model')
model = Sequential()
model.add(Conv2D(64,
                 (3,3),
                 activation = 'relu',
                 kernel_initializer = 'he_uniform',
                 padding = 'same',
                 input_shape = (200, 200, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,
                 (3,3),
                 activation = 'relu',
                 kernel_initializer = 'he_uniform',
                 padding = 'same'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
# model.add(Dense(10, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(Dense(1, activation = 'sigmoid'))
print('Compiling model')
model.compile(
              Adam(lr = 0.5),
              # SGD(lr = 0.1, momentum = 0.0),
              loss = 'binary_crossentropy',
              metrics = ['accuracy']
             )
# print('Fitting data with compiled model')
# model.fit(X_train, y_train, epochs = 5, batch_size = 10, verbose = 2)
#
# print('Evaluating model')
# loss, acc = model.evaluate(X_test, y_test, verbose = 0)
# print(loss, acc)
