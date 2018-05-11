from __future__ import division, print_function, absolute_import

import datetime
import os
import random as rand

import cv2
import numpy as np
import tensorflow as tf
import tflearn
from cv2.cv2 import imread
from numpy import array
from scipy._lib.six import xrange
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import shuffle
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from utils import printProgressBar

CLASSIFIER_FILE_TFL = "eye-veins-classifier.tfl"

max_height = 600
MASK_SIZE = 21
k = 10
PHOTO_SAMPLES = 8000


def chunkify(lst, n):
    return [lst[i::n] for i in xrange(n)]


class Load:
    root_dir = "all/"

    def __init__(self, path):
        self.data = []
        self.path = path

    def threshold(self, value=10):
        for mask in self.data:
            mask[mask > value] = 255
            mask[mask <= value] = 0

    @staticmethod
    def load_image(file_name):
        img = imread(file_name)[:, :, 1]
        current_height = img.shape[1]
        if current_height > max_height:
            max_width = int(max_height / current_height * img.shape[0])
            img = cv2.resize(img, dsize=(max_height, max_width), interpolation=cv2.INTER_CUBIC)
        return img.astype(int)

    def load_all(self):
        self.data = [Load.load_image(Load.root_dir + self.path + "/" + p)
                     for p in os.listdir(Load.root_dir + self.path)]


class LearnData:
    def __init__(self) -> None:
        self.images = Load("images")
        self.masks = Load("mask")
        self.manual = Load("manual1")

    def load_all(self):
        for l in [self.images, self.manual, self.masks]:
            l.load_all()
        self.masks.threshold()

    def zip(self):
        return zip(self.images.data, self.manual.data, self.masks.data)

    @staticmethod
    def get_possible_points(mask):
        indexes = np.where(mask > 0)
        all_x, all_y = indexes
        indexes = [[all_x[i], all_y[i]] for i in range(0, len(all_x))]
        max_x = mask.shape[0] - MASK_SIZE - 5
        max_y = mask.shape[1] - MASK_SIZE - 5
        return [i for i in indexes if 5 < i[0] < max_x and 5 < i[1] < max_y]

    @staticmethod
    def normalize(l):
        return list(array(l) / 255)

    def prepare_learn_data(self):
        X = []
        Y = []
        for image, manual, mask in self.zip():
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

    def create_model(self, name="InputData", ckpt=None):
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
        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.4)
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        self.model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path=ckpt)

    def train(self, X, Y, Xtest, Ytest):
        self.model.fit(X, Y, n_epoch=8, shuffle=True, validation_set=(Xtest, Ytest),
                       show_metric=True, batch_size=64, snapshot_epoch=True)

    def load(self, path):
        self.model.load(path)

    def predict(self, toPredict):
        return self.model.predict([toPredict])

    def save(self, file_name):
        self.model.save(file_name)

    def mark(self, img):
        reconstructed = np.zeros((img.shape[0], img.shape[1]))
        if img.max() > 1:
            img = img / 255
        max_x = img.shape[0] - MASK_SIZE - 1
        for x in range(0, max_x):
            printProgressBar(x, max_x, prefix='Progress:', suffix='Complete', length=50)

            for y in range(0, img.shape[1] - MASK_SIZE - 1):
                centerX = x + int(MASK_SIZE / 2)
                centerY = y + int(MASK_SIZE / 2)
                sample = img[x:(x + MASK_SIZE), y:(y + MASK_SIZE)]
                sample = array(sample).reshape(MASK_SIZE, MASK_SIZE, 1)
                prediction = self.predict(sample)
                result = prediction.T[0]

                if result > 0.4:
                    reconstructed[centerX][centerY] = 1
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


def run_session(splittedSamples, splittedMasks):
    X = [item for sublist in splittedSamples for item in sublist]
    Y = [item for sublist in splittedMasks for item in sublist]
    Xtest = splittedSamples.pop(i)
    Ytest = splittedMasks.pop(i)
    network = Network()
    network.create_model(name='InputData' + str(i))
    network.train(X, Y, Xtest, Ytest)

    accuracy = get_accuracy(network, Xtest, Ytest) / len(Ytest)
    print("Accuracy: " + str(accuracy))
    network.save("result/" + str(accuracy) + " " + CLASSIFIER_FILE_TFL)
    print("Network trained and saved as %s!" % CLASSIFIER_FILE_TFL)
    tf.reset_default_graph()
    return accuracy


if __name__ == '__main__':
    learn_data = LearnData()
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
        averageAcc += run_session(splittedSamples, splittedMasks)
    averageAcc = averageAcc / k
    print(averageAcc)
    text_file = open("result/averageACC.txt", "w")
    text_file.write("Average accuracy: " + str(averageAcc))
    text_file.close()
