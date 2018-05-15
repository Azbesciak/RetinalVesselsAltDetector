from __future__ import division, print_function, absolute_import

import codecs

from convolution_network import *
from utils import TEST_PATH, NETWORK_RESULT_DIR, IMG_PROC_RESULT_DIR

VALIDATION_README_MD = TEST_PATH + "/validation/README.MD"


def compare_all(label, learn_data: LearnData, to_compare: Load):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total_points_num = 0
    for manual, reconstructed, mask, org in zip(learn_data.manual.get_data(),
                                                to_compare.get_data(),
                                                learn_data.masks.get_data(),
                                                learn_data.original.images):
        file_name = org.get_file_name()
        result, FN, FP, TN, TP, points_num = compare(manual, mask, reconstructed, FN, FP, TN, TP)
        total_points_num += points_num
        Load.save(f"{TEST_PATH}/validation/{label}/{file_name}", result)

    return get_metric(FN, FP, TN, TP, label, total_points_num)


def f(v):
    return "{0:.6f}".format(round(v, 6))


def get_metric(FN, FP, TN, TP, label, total_points_num):
    P = TP + FN
    N = TN + FP
    RATIO = P / N
    ACC = (TP + TN) / (P + N)
    ERR = (FP + FN) / (P + N)
    TPR = TP / (TP + FN)
    TNP = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    MSE = (TP + FN) / total_points_num
    N_ACC = (TP + TN * RATIO) / (P + N * RATIO)
    N_ERR = (FP + FN * RATIO) / (P + N * RATIO)
    N_TPR = TP / (TP + FN * RATIO)
    N_TNP = TN / (TN + FP * RATIO)
    return f"""## {label}
           | Measure | Value |
           | --- | --- |
           | Accuracy |{f(ACC)}|
           | Error level |{f(ERR)}|
           | Sensitivity | {f(TPR)} |
           | Specificity | {f(TNP)} |
           | average from Specificity and Sensitivity| {f(np.sqrt(TNP*TPR))} |
           | Positive Predictive Value | {f(PPV)} |
           | Negative Predictive Value | {f(NPV)} |
           | Medium Square Error | {f(MSE)} |
           | Positive to negative value in original | {f(RATIO)} |
           | Accuracy(Norm) | {f(N_ACC)} |
           | Error level(Norm) | {f(N_ERR)} |
           | Sensitivity(Norm) | {f(N_TPR)} |
           | Specificity(Norm) | {f(N_TNP)} |"""


def compare(manual, mask, reconstructed, fn=0, fp=0, tn=0, tp=0):
    reconstructed[reconstructed > 10] = 255
    reconstructed[reconstructed <= 10] = 0
    if reconstructed.shape != manual.shape:
        print("SHAPES ARE INVALID")
    points = LearnData.get_possible_points(mask)
    points = array(points) + MASK_SIZE // 2
    result = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for x, y in points:
        if reconstructed[x][y] == 255 and manual[x][y] == 255:
            tp += 1
            result[x][y] = [46, 125, 50]
        elif reconstructed[x][y] == 255 and manual[x][y] == 0:
            fp += 1
            result[x][y] = [239, 20, 87]
        elif reconstructed[x][y] == 0 and manual[x][y] == 255:
            fn += 1
            result[x][y] = [239, 154, 154]
        elif reconstructed[x][y] == 0 and manual[x][y] == 0:
            tn += 1
            result[x][y] = [174, 213, 129]
        else:
            result[x][y] = [0, 0, 255]
    return result, fn, fp, tn, tp, len(points)


if __name__ == '__main__':
    learn_data = LearnData(TEST_PATH)
    learn_data.load_all()

    neural_res = Load(NETWORK_RESULT_DIR, TEST_PATH)
    neural_res.load_all()
    image_proc = Load(IMG_PROC_RESULT_DIR, TEST_PATH)
    image_proc.load_all()
    header = """> Meaning:\\
    > Green - correct\\
    > Red - incorrect\\
    > Blue - ???\\
    > Violet - both close to each other... green + red\\
    > Light - everything else but not vessel\\
    > Dark - vessel\\"""
    neural_stat = compare_all(NETWORK_RESULT_DIR, learn_data, neural_res)
    image_stat = compare_all(IMG_PROC_RESULT_DIR, learn_data, image_proc)
    with codecs.open(VALIDATION_README_MD, "w", "utf-8-sig") as f:
        f.write(f"{header}\n{neural_stat}\n{image_stat}")
