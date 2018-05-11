# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from convolution_network import Network, CLASSIFIER_FILE_TFL, LearnData
from utils import Load, TEST_PATH

imageName = "test2.jpg"
model_path = "model/"

if __name__ == '__main__':
    network = Network()
    network.create_model(ckpt='eye-veins.tfl.ckpt')
    network.load(model_path + CLASSIFIER_FILE_TFL)
    data = LearnData(TEST_PATH)
    data.load_all()
    for mask, org in zip(data.masks.images, data.original.images):
        reconstructed = network.mark(org.image, mask.image)
        Load.save(TEST_PATH + "/network/" + org.get_file_name(), reconstructed)
        exit(0)
