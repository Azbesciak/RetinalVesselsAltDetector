# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from skimage import data as img
import convolution_network
img_path = "test"
img_name = ""
directory = os.getcwd() + '/'
# Load the image file
reconstructed = np.load("plik2.npy")
reconstructed = [i * 255 for i in reconstructed]
reconstructed = array(reconstructed)
mask = array(img.load(directory + "test2.tif", True))
for i in range(len(mask)):
    for j in range(len(mask[i])):
        if mask[i][j] > 10:
            mask[i][j] = 255
        else:
            mask[i][j] = 0
TP = 0
TN = 0
FP = 0
FN = 0
result = np.zeros((reconstructed.shape[0], reconstructed.shape[1], 3), dtype=np.uint8)
for x in range(0, reconstructed.shape[0] - convolution_network.MASK_SIZE - 1):
    for y in range(0, reconstructed.shape[1] - convolution_network.MASK_SIZE.MASK_SIZE - 1):
        if reconstructed[x][y] == 255 and mask[x][y] == 255:
            TP += 1
            result[x][y] = [127, 255, 0]
        elif reconstructed[x][y] == 255 and mask[x][y] == 0:
            FP += 1
            result[x][y] = [255, 0, 0]
        elif reconstructed[x][y] == 0 and mask[x][y] == 255:
            FN += 1
            result[x][y] = [255, 0, 0]
        elif reconstructed[x][y] == 0 and mask[x][y] == 0:
            TN += 1
            result[x][y] = [127, 255, 0]
        else:
            print(reconstructed[x][y])
            print(mask[x][y])
            result[x][y] = [0, 0, 255]

P = TP + FN
N = TN + FP
ACC = (TP + TN) / (P + N)
ERR = (FP + FN) / (P + N)
TPR = TP / (TP + FN)  # czułość
TNP = TN / (TN + FP)  # specyficzność
PPV = TP / (TP + FP)  # precyzja przewidywania pozytywnego
NPV = TN / (TN + FN)  # precyzja przewidywania negatywnego
print("Dokładność: " + str(ACC))
print("Poziom błędu: " + str(ERR))
print("Czułość: " + str(TPR))
print("Specyficzność: " + str(TNP))
print("PPP: " + str(PPV))
print("PPN: " + str(NPV))
np.save("validationImage", result)
plt.imshow(array(result))
cv2.imwrite("filename.jpg", reconstructed)
plt.show()
