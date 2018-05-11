# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import cv2
import matplotlib.pyplot as plt

from convolution_network import Network, Load

test_path = "test/"
imageName = "test2.jpg"
model_path = "model/"

network = Network()
network.create_model(ckpt='eye-veins.tfl.ckpt')
network.load(model_path + "eye-veins-classifier.tfl")
img = Load.load_image(test_path + imageName)
reconstructed = network.mark(img)
cv2.imwrite(test_path + "res_" + imageName, reconstructed)
plt.imshow(reconstructed)
plt.show()
