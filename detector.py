import matplotlib.pyplot as plt
from cv2.cv2 import imread

from image_processing import process

src_dir = "all/"

img_name = "images/01_dr.JPG"


def draw_image(img):
    plt.subplot(1, 1, 1)
    plt.imshow(img, cmap=plt.cm.Greys_r)


def get_img():
    return imread(src_dir + img_name)


if __name__ == '__main__':
    img = get_img()
    res = process(img)
    draw_image(res)
    plt.show()
