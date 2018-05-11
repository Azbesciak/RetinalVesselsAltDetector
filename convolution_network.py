from __future__ import division, print_function, absolute_import

import datetime
import random as rand

import numpy as np
import tensorflow as tf
import tflearn
from numpy import array
from scipy._lib.six import xrange
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import shuffle
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from utils import Load, printProgressBar, LEARNING_PATH

N_EPOCH = 10

MODEL_OUTPUT = "model"

KEEP_PROB = 0.6
CONNECTIONS = 1024
CLASSIFIER_FILE_TFL = "eye-veins-classifier.tfl"

MASK_SIZE = 25
k = 10
PHOTO_SAMPLES = 30000
CHUNK_STEP = 5


def chunkify(lst, n):
    return [lst[i::n] for i in xrange(n)]


class LearnData:
    def __init__(self, root_dir) -> None:
        self.original = Load("images", root_dir)
        self.masks = Load("mask", root_dir)
        self.manual = Load("manual1", root_dir)

    def load_all(self):
        for l in [self.original, self.manual, self.masks]:
            l.load_all()
        self.masks.threshold()
        self.manual.threshold()

    def zip_data(self):
        return zip(self.original.get_data(), self.manual.get_data(), self.masks.get_data())

    @staticmethod
    def get_possible_points(mask):
        def to_corner(v):
            return v - MASK_SIZE // 2 - 1

        indexes = np.where(mask > 0)
        all_x, all_y = indexes
        indexes = [[to_corner(all_x[i]), to_corner(all_y[i])] for i in range(0, len(all_x))]
        max_x = mask.shape[0] - MASK_SIZE - 1
        max_y = mask.shape[1] - MASK_SIZE - 1
        return [i for i in indexes if 0 <= i[0] < max_x and 0 <= i[1] < max_y]

    @staticmethod
    def normalize(l):
        return list(array(l) / 255)

    def prepare_learn_data(self):
        X = []
        Y = []
        for image, manual, mask in self.zip_data():
            possible_points = LearnData.get_possible_points(mask)
            max_index = len(possible_points) - 1
            for i in range(0, PHOTO_SAMPLES):
                result, sample = self.get_sample(image, manual, max_index, possible_points)
                X.append(sample)
                Y.append(result)
        X = LearnData.normalize(X)
        Y = LearnData.normalize(Y)
        return shuffle(X, Y)

    @staticmethod
    def get_sample(image, manual, max_index, possible_points):
        flag = -1
        vein_flag = rand.randint(0, 2)
        while flag != vein_flag:
            start_x, start_y = possible_points[rand.randint(0, max_index)]
            end_x = start_x + MASK_SIZE
            end_y = start_y + MASK_SIZE
            center_x = start_x + int(MASK_SIZE / 2)
            center_y = start_y + int(MASK_SIZE / 2)
            sample = image[start_x:end_x, start_y:end_y]
            sample = array(sample).reshape(MASK_SIZE, MASK_SIZE, 1)
            value_in_center = manual[center_x, center_y]
            if value_in_center == 255:
                result = [1, 0]
                flag = rand.randint(1, 2)
            else:
                result = [0, 1]
                flag = 0
        return result, sample


class Network:
    def __init__(self):
        self.model = None

    def create_model(self, name="InputData"):
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()
        network = input_data(shape=[MASK_SIZE, MASK_SIZE, 1],
                             data_preprocessing=img_prep, name=name)
        print(network)
        network = conv_2d(network, 32, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, CONNECTIONS, activation='relu')
        network = dropout(network, KEEP_PROB)
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        self.model = tflearn.DNN(network, tensorboard_verbose=0)

    def train(self, X, Y, Xtest, Ytest):
        self.model.fit(X, Y, n_epoch=N_EPOCH, shuffle=True, validation_set=(Xtest, Ytest),
                       show_metric=True, batch_size=64, snapshot_epoch=True)

    def load(self, path):
        self.model.load(path)

    def predict(self, toPredict):
        return self.model.predict([toPredict])

    def save(self, file_name):
        self.model.save(file_name)

    def mark(self, img, mask=None):
        reconstructed = np.zeros((img.shape[0], img.shape[1]))
        if img.max() > 1:
            img = img / 255
        if mask is not None:
            points = LearnData.get_possible_points(mask)
        else:
            points = [[x, y]
                      for x in range(0, img.shape[0] - MASK_SIZE - 1)
                      for y in range(0, img.shape[1] - MASK_SIZE - 1)
                      ]
        total = len(points)
        for i, p in enumerate(points):
            printProgressBar(i, total, prefix='Progress:', suffix='Complete', length=100)
            x, y = p
            centerX = x + int(MASK_SIZE / 2)
            centerY = y + int(MASK_SIZE / 2)
            sample = img[x:(x + MASK_SIZE), y:(y + MASK_SIZE)]
            sample = array(sample).reshape(MASK_SIZE, MASK_SIZE, 1)
            prediction = self.predict(sample)
            result = prediction.T[0]

            if result > KEEP_PROB:
                reconstructed[centerX][centerY] = 255
            else:
                reconstructed[centerX][centerY] = 0

        return array(reconstructed)


def get_accuracy(network: Network, Xtest, Ytest):
    accuracy = 0
    for toPredict, actual in zip(Xtest, Ytest):
        prediction = network.predict(toPredict)
        predicted_class = np.argmax(prediction)
        actual_class = np.argmax(actual)
        if predicted_class == actual_class:
            accuracy += 1
    return accuracy


def run_session(splittedSamples, splittedMasks, i):
    X = [item for sublist in splittedSamples for item in sublist]
    Y = [item for sublist in splittedMasks for item in sublist]
    Xtest = list(splittedSamples).pop(i)
    Ytest = list(splittedMasks).pop(i)
    network = Network()
    network.create_model(name='InputData' + str(i))
    network.train(X, Y, Xtest, Ytest)

    accuracy = get_accuracy(network, Xtest, Ytest) / len(Ytest)
    print("Accuracy: " + str(accuracy))
    network.save(MODEL_OUTPUT + "/" + str(accuracy) + " " + CLASSIFIER_FILE_TFL)
    print("Network trained and saved as %s!" % CLASSIFIER_FILE_TFL)
    tf.reset_default_graph()
    return accuracy


if __name__ == '__main__':
    learn_data = LearnData(LEARNING_PATH)
    learn_data.load_all()
    X, Y = learn_data.prepare_learn_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    splittedSamples = chunkify(X, k)
    splittedMasks = chunkify(Y, k)

    timeStamp = datetime.datetime.now().time()

    averageAcc = 0
    for i in range(0, k):
        averageAcc += run_session(splittedSamples, splittedMasks, i)
    averageAcc = averageAcc / k
    print(averageAcc)
    text_file = open(MODEL_OUTPUT + "/averageACC.txt", "w")
    text_file.write("Average accuracy: " + str(averageAcc))
    text_file.close()
