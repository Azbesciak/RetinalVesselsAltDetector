from __future__ import division, print_function, absolute_import

import codecs

from convolution_network import *
from utils import TEST_PATH, NETWORK_RESULT_DIR, IMG_PROC_RESULT_DIR

VALIDATION_README_MD = TEST_PATH + "/validation/README.MD"


def get_stat(label, learn_data : LearnData, to_compare : Load):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total_points_num = 0
    for manual, reconstructed, mask, org in zip(learn_data.manual.get_data(),
                                           to_compare.get_data(),
                                           learn_data.masks.get_data(),
                                           learn_data.original.images):
        reconstructed[reconstructed > 10] = 255
        reconstructed[reconstructed <= 10] = 0
        if reconstructed.shape != manual.shape:
            print("SHAPES ARE INVALID")
        points = LearnData.get_possible_points(mask)
        total_points_num += len(points)
        file_name = org.get_file_name()
        result = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x, y in points:
            if reconstructed[x][y] == 255 and manual[x][y] == 255:
                TP += 1
                result[x][y] = [46, 125, 50]
            elif reconstructed[x][y] == 255 and manual[x][y] == 0:
                FP += 1
                result[x][y] = [239, 20, 87]
            elif reconstructed[x][y] == 0 and manual[x][y] == 255:
                FN += 1
                result[x][y] = [239, 154, 154]
            elif reconstructed[x][y] == 0 and manual[x][y] == 0:
                TN += 1
                result[x][y] = [174, 213, 129]
            else:
                result[x][y] = [0, 0, 255]
        Load.save(TEST_PATH + "/validation/" + label + "/" + file_name, result)

    P = TP + FN
    N = TN + FP
    RATIO = P/N
    ACC = (TP + TN) / (P + N)
    ERR = (FP + FN) / (P + N)
    TPR = TP / (TP + FN)
    TNP = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    MSE = (TP + FN) / total_points_num

    N_ACC = (TP + TN*RATIO) / (P + N*RATIO)
    N_ERR = (FP + FN*RATIO) / (P + N*RATIO)
    N_TPR = TP / (TP + FN*RATIO)
    N_TNP = TN / (TN + FP*RATIO)
    with codecs.open(VALIDATION_README_MD, "a", "utf-8-sig") as f:
        f.write("\n## " + label + "\n")
        f.write("| Measure | Value |\n")
        f.write("| --- | --- |\n")
        f.write("| Accuracy |" + str(ACC) + "|\n")
        f.write("| Error level |" + str(ERR) + "|\n") 
        f.write("| Sensitivity | " + str(TPR) + " |\n")
        f.write("| Specificity | " + str(TNP) + " |\n")
        f.write("| Positive Predictive Value | " + str(PPV) + " |\n")
        f.write("| Negative Predictive Value | " + str(NPV) + " |\n")
        f.write("| Medium Square Error | " + str(MSE) + "\n")
        f.write("| Positive to negative value in original | " + str(RATIO) + " |\n")
        f.write("| Normalized results with ratio | " + str(RATIO) + " |\n")
        f.write("| Accuracy | " + str(N_ACC) + " |\n")
        f.write("| Error level | " + str(N_ERR) + " |\n")
        f.write("| Sensitivity | " + str(N_TPR) + " |\n")
        f.write("| Specificity | " + str(N_TNP) + " |\n")


if __name__ == '__main__':
    learn_data = LearnData(TEST_PATH)
    learn_data.load_all()

    neural_res = Load(NETWORK_RESULT_DIR, TEST_PATH)
    neural_res.load_all()
    image_proc = Load(IMG_PROC_RESULT_DIR, TEST_PATH)
    image_proc.load_all()
    with open(VALIDATION_README_MD, 'w') as f:
        f.write("# Results comparision\n")
        f.write("> Meaning:\\\n")
        f.write("> Green - correct\\\n")
        f.write("> Red - incorrect\\\n")
        f.write("> Blue - ???\\\n")
        f.write("> Violet - both close to each other... green + red\\\n")
        f.write("> Light - everything else but not vessel\\\n")
        f.write("> Dark - vessel\\\n")
    get_stat(NETWORK_RESULT_DIR, learn_data, neural_res)
    get_stat(IMG_PROC_RESULT_DIR, learn_data, image_proc)
