# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from convolution_network import Network, CLASSIFIER_FILE_TFL, LearnData
from utils import Load, TEST_PATH, NETWORK_RESULT_DIR

model_path = "model/"


def create_network(progress_length=100):
    network = Network(progress_length)
    network.create_model()
    network.load(model_path + CLASSIFIER_FILE_TFL)
    return network


if __name__ == '__main__':
    network = create_network()
    data = LearnData(TEST_PATH)
    data.load_all()
    for mask, org in zip(data.masks.images, data.original.images):
        reconstructed = network.mark(org.image, mask.image, org.get_file_name())
        Load.save(TEST_PATH + "/" + NETWORK_RESULT_DIR + "/" + org.get_file_name(), reconstructed)
