import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

PADDING = 25 // 2 + 1

MAX_IMG_WIDTH = 600
TEST_PATH = "test"
LEARNING_PATH = "all"
NETWORK_RESULT_DIR = "network"
IMG_PROC_RESULT_DIR = "imgproc"

OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
ENDC = '\033[0m'


# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    iteration_step = int(total / 1000)
    if iteration % iteration_step != 0:
        return
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(('\r' + OKBLUE + '%s' + ENDC + ' |' + OKGREEN + '%s' + ENDC + '| %s%% %s\x1b[3A') % (
        prefix, bar, percent, suffix), end="", flush=True)
    # Print New Line on Complete
    if iteration == total:
        print(" ")


class Img:
    def __init__(self, name, image):
        self.name = name
        self.image = image

    def get_file_name(self):
        return os.path.basename(self.name)


class Load:
    def __init__(self, path, root_dir):
        self.images = []
        self.path = path
        self.root_dir = root_dir

    def get_data(self):
        return [i.image for i in self.images]

    def threshold(self, value=10):
        for mask in self.get_data():
            mask[mask > value] = 255
            mask[mask <= value] = 0

    @staticmethod
    def save(path, img):
        cv2.imwrite(path, img)

    @staticmethod
    def load_image(file_name):
        img = cv2.imread(file_name)[:, :, 1]
        current_width = img.shape[1]
        if current_width > MAX_IMG_WIDTH:
            max_height = int(MAX_IMG_WIDTH / current_width * img.shape[0])
            img = cv2.resize(img, dsize=(MAX_IMG_WIDTH, max_height), interpolation=cv2.INTER_CUBIC)

        img = np.pad(img, ((PADDING, PADDING), (PADDING, PADDING)), 'constant', constant_values=0)
        return Img(file_name, img)

    def load_all(self):
        self.images = [Load.load_image(self.root_dir + "/" + self.path + "/" + p)
                       for p in os.listdir(self.root_dir + "/" + self.path)]


def draw_image(img):
    plt.subplot(1, 1, 1)
    plt.imshow(img, cmap=plt.cm.Greys_r)
    plt.show()
