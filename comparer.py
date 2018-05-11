from __future__ import division, print_function, absolute_import

import codecs

from convolution_network import *
from utils import TEST_PATH, NETWORK_RESULT_DIR, IMG_PROC_RESULT_DIR


def get_stat(label, learn_data: LearnData, to_compare: Load):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total_points_num = 0
    for manual, reconstructed, mask, org in zip(learn_data.manual.get_data(),
                                           to_compare.get_data(),
                                           learn_data.masks.get_data(),
                                           learn_data.original.images):
        points = LearnData.get_possible_points(mask)
        total_points_num += len(points)
        file_name = org.get_file_name()
        result = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x, y in points:
            if reconstructed[x][y] == 255 and manual[x][y] == 255:
                TP += 1
                result[x][y] = [127, 255, 0]
            elif reconstructed[x][y] == 255 and manual[x][y] == 0:
                FP += 1
                result[x][y] = [255, 0, 0]
            elif reconstructed[x][y] == 0 and manual[x][y] == 255:
                FN += 1
                result[x][y] = [255, 0, 0]
            elif reconstructed[x][y] == 0 and manual[x][y] == 0:
                TN += 1
                result[x][y] = [127, 255, 0]
            else:
                result[x][y] = [0, 0, 255]
        Load.save(TEST_PATH + "/validation/" + label + "/" + file_name, result)

    P = TP + FN
    N = TN + FP
    ACC = (TP + TN) / (P + N)
    ERR = (FP + FN) / (P + N)
    TPR = TP / (TP + FN)  # czułość
    TNP = TN / (TN + FP)  # specyficzność
    PPV = TP / (TP + FP)  # precyzja przewidywania pozytywnego
    NPV = TN / (TN + FN)  # precyzja przewidywania negatywnego
    MSE = (TP + FN) / total_points_num  # średni błąd
    with codecs.open(TEST_PATH + "/validation/" + label + ".txt", "w", "utf-8-sig") as f:
        f.write("Dokładność: " + str(ACC) + "\n")
        f.write("Poziom błędu: " + str(ERR) + "\n")
        f.write("Czułość: " + str(TPR) + "\n")
        f.write("Specyficzność: " + str(TNP) + "\n")
        f.write("PPP: " + str(PPV) + "\n")
        f.write("PPN: " + str(NPV) + "\n")
        f.write("MSE:" + str(MSE) + "\n")


if __name__ == '__main__':
    learn_data = LearnData(TEST_PATH)
    learn_data.load_all()

    neural_res = Load(NETWORK_RESULT_DIR, TEST_PATH)
    neural_res.load_all()
    image_proc = Load(IMG_PROC_RESULT_DIR, TEST_PATH)
    image_proc.load_all()
    get_stat(NETWORK_RESULT_DIR, learn_data, neural_res)
    get_stat(IMG_PROC_RESULT_DIR, learn_data, image_proc)
