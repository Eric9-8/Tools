# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/10/26 20:21
import cv2.cv2 as cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def img_pre(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.medianBlur(img, 9)
    return img


def grayHist(img):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    return pixelSequence


def hist_pre(img_gray):
    img_enhance = cv2.equalizeHist(img_gray)
    img_enhance = cv2.GaussianBlur(img_enhance, (9, 9), sigmaX=0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    lmt_img_enhance = clahe.apply(img_gray)
    lmt_img_enhance = cv2.GaussianBlur(lmt_img_enhance, (9, 9), sigmaX=0)

    number_bins = 256
    pix1 = grayHist(img_gray)
    pix2 = grayHist(img_enhance)
    pix3 = grayHist(lmt_img_enhance)

    plt.subplot(2, 3, 1)
    plt.imshow(img_gray, cmap="gray", vmin=0, vmax=255), plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(img_enhance, cmap="gray", vmin=0, vmax=255), plt.title('Gray-hist Image')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(lmt_img_enhance, cmap="gray", vmin=0, vmax=255), plt.title('Limited-Gray-hist Image')
    plt.axis('off')

    ax1 = plt.subplot(2, 3, 4)
    plt.hist(pix1, number_bins, facecolor='gray', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    ax1.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.xlim([0, 255])

    ax2 = plt.subplot(2, 3, 5)
    plt.hist(pix2, number_bins, facecolor='gray', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    ax2.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.xlim([0, 255])

    ax3 = plt.subplot(2, 3, 6)
    plt.hist(pix3, number_bins, facecolor='gray', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    ax3.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.xlim([0, 255])

    plt.show()
    return img_gray, img_enhance, lmt_img_enhance


def sobel_operator(img):
    # print(img.shape)
    sbl_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=9)
    sbl_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=9)

    return sbl_horizontal, sbl_vertical


def Lap_operator(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return lap


def canny_operator(img):
    # img = cv2.medianBlur(img, 11)
    img = cv2.GaussianBlur(img, (7, 7), sigmaX=0)
    canny = cv2.Canny(img, 100, 105)
    return canny


if __name__ == '__main__':
    path = 'D:/Mobilenet_CoordinateAttention_GAF/8.jpg'
    img = img_pre(path)
    img1, img2, img3 = hist_pre(img)

    # ------------------sobel operator--------------------- #

    sob_h_1, sob_v_1 = sobel_operator(img1)
    sob_h_2, sob_v_2 = sobel_operator(img2)
    sob_h_3, sob_v_3 = sobel_operator(img3)

    plt.subplot(1, 6, 1)
    plt.imshow(sob_h_1, cmap='gray'), plt.axis('off'), plt.title("sobel horizontal original")
    plt.subplot(1, 6, 2)
    plt.imshow(sob_v_1, cmap='gray'), plt.axis('off'), plt.title("sobel vertical original")
    plt.subplot(1, 6, 3)
    plt.imshow(sob_h_2, cmap='gray'), plt.axis('off'), plt.title("sobel horizontal Global hist ")
    plt.subplot(1, 6, 4)
    plt.imshow(sob_v_2, cmap='gray'), plt.axis('off'), plt.title("sobel vertical Global hist")
    plt.subplot(1, 6, 5)
    plt.imshow(sob_h_3, cmap='gray'), plt.axis('off'), plt.title("sobel horizontal Adaptive hist")
    plt.subplot(1, 6, 6)
    plt.imshow(sob_v_3, cmap='gray'), plt.axis('off'), plt.title("sobel vertical Adaptive hist")

    plt.show()

    # ------------------Laplace operator--------------------- #

    laplace_1 = Lap_operator(img1)
    laplace_2 = Lap_operator(img2)
    laplace_3 = Lap_operator(img3)

    plt.subplot(1, 3, 1)
    plt.imshow(laplace_1, cmap='gray'), plt.axis('off'), plt.title("Laplace original")
    plt.subplot(1, 3, 2)
    plt.imshow(laplace_2, cmap='gray'), plt.axis('off'), plt.title("Laplace Global hist")
    plt.subplot(1, 3, 3)
    plt.imshow(laplace_3, cmap='gray'), plt.axis('off'), plt.title("Laplace Adaptive hist ")

    plt.show()

    # ------------------canny operator--------------------- #

    canny_1 = canny_operator(img1)
    canny_2 = canny_operator(img2)
    canny_3 = canny_operator(img3)

    plt.subplot(1, 3, 1)
    plt.imshow(canny_1, cmap='gray'), plt.axis('off'), plt.title("Canny original")
    plt.subplot(1, 3, 2)
    plt.imshow(canny_2, cmap='gray'), plt.axis('off'), plt.title("Canny Global hist")
    plt.subplot(1, 3, 3)
    plt.imshow(canny_3, cmap='gray'), plt.axis('off'), plt.title("Canny Adaptive hist ")

    plt.show()
