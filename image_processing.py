import cv2
import numpy as np


def extract_green(img):
    return img[:, :, 1]


def process(img):
    if len(img.shape) == 3:
        img = extract_green(img)
    f5 = show_vessels(img)
    vessels_without_noise = remove_noise_from_vessels(f5)
    blood_vessels = get_long_vessels(vessels_without_noise, img)
    return blood_vessels


def get_long_vessels(fundus_eroded, green_fundus):
    xmask = get_mask(green_fundus)
    x1, xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and 3000 >= cv2.contourArea(cnt) >= 100:
            cv2.drawContours(xmask, [cnt], -1, 0, -1)
    blood_vessels = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
    return blood_vessels


def get_mask(green_fundus):
    return np.ones(green_fundus.shape[:2], dtype="uint8") * 255


def remove_noise_from_vessels(f5):
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    im2, contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    fundus_eroded = cv2.bitwise_not(newfin)
    return fundus_eroded


def show_vessels(green):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_green_fundus = clahe.apply(green)
    r = open_and_close(contrast_enhanced_green_fundus, 5)
    r = open_and_close(r, 13)
    r = open_and_close(r, 27)
    without_original = cv2.subtract(r, contrast_enhanced_green_fundus)
    increased = clahe.apply(without_original)
    return increased


def open_and_close(img, radius):
    o1 = cv2.morphologyEx(img, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)), iterations=1)
    return cv2.morphologyEx(o1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)),
                            iterations=1)