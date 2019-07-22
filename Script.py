from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import glob

# LABELS NOTE:
#       Dog: 1
#       Cat: 0

def visualize_convolutions(ncols, nrows):
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())

    transposed_conv_weight_matrix = np.reshape(np.transpose(np.array(weights[0][0])), (ncols*nrows, 10, 10))
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows)

    for i in range(transposed_conv_weight_matrix.shape[0]):
        axes.ravel()[i].imshow(transposed_conv_weight_matrix[i], cmap='gray', interpolation='nearest')

    plt.show()


def randomize_and_split(labels, examples):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    data_counts = [0,0]

    print("Randomizing data")
    balanced_examples_num = len(examples) * 0.8 / 2.0

    while examples:
        rand = random.randint(0, len(examples) - 1)

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

    return [np.subtract(np.asarray(X_train), 127.5), np.subtract(np.asarray(X_test), 127.5), np.asarray(y_train), np.asarray(y_test)]


def shape_data(example):
    shaped_example = []
    count = 0

    for i in range(200):
        row = []
        for j in range(200):
            row.append([float(example[count])])
            count += 1

        shaped_example.append(row)
    return shaped_example

def build_model():
    print('Creating Model')
    model = Sequential()
    # model.add(Conv2D(32,
    #                  (10,10),
    #                  activation = 'relu',
    #                  kernel_initializer = 'he_uniform',
    #                  padding = 'same',
    #                  input_shape = (200, 200, 1)))
    # model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(16,
                     (5,5),
                     activation = 'sigmoid',
                     kernel_initializer = 'he_uniform',
                     padding = 'same'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(8,
                     (5,5),
                     activation = 'sigmoid',
                     kernel_initializer = 'he_uniform',
                     padding = 'same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    # model.add(Dense(20, activation = 'sigmoid', kernel_initializer = 'he_uniform'))
    model.add(Dense(20, activation = 'sigmoid', kernel_initializer = 'he_uniform'))
    model.add(Dense(1, activation = 'sigmoid'))
    print('Compiling model')
    model.compile(
                  # Adam(lr = 0.01),
                  SGD(lr = 0.01, momentum = 0.0),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy']
                 )
    return model


model = build_model()
files = glob.glob("../Data/DogsVsCats/grayscale_csv_sets_10/*.csv")
# files = glob.glob("../Data/DogsVsCats/colored_csv_sets/*.csv")

for set in files:
    examples = []
    labels = []

    with open(set) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        print("Reshaping data")
        for row in csv_reader:
            labels.append(int(row[0]))
            examples.append(shape_data(row[1:]))

    randomized_data = randomize_and_split(labels, examples)
    X_train = randomized_data[0]
    X_test = randomized_data[1]
    y_train = randomized_data[2]
    y_test = randomized_data[3]

    model.fit(X_train, y_train, epochs = 5, batch_size = 10, verbose = 1)

    loss,acc =  model.evaluate(X_test, y_test, verbose = 0)
    print("Loss:", loss, "\nAccuracy:", acc)
    # visualize_convolutions(4,4)

# TODO: find a way to optimize the time complexity on reshaping and randomizing
# TODO: find a way to have a timer for the user, wait 5 seconds to see if they want to continue, continue if no response comes in
    # response = input("Train with another set?\n")
    # type(response)
    #
    # if 'y' in response:
    #     del labels
    #     del examples
    #     pass
    # else:
    #     break

# After finishing processing data, figure out how to export the model for comparison with others
