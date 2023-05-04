# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/10/26 20:14
import math
import cv2.cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

# load an original image

img = cv2.imread('D:/Pytorch/image_test/image/3.jpg')

# color value range
cRange = 256

rows, cols, channels = img.shape

# convert color space from bgr to gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# laplacian edge
imgLap = cv2.Laplacian(imgGray, cv2.CV_8U)

# otsu method
threshold, imgOtsu = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# adaptive gaussian threshold
imgAdapt = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# imgAdapt = cv2.medianBlur(imgAdapt, 3)

# display original image and gray image
plt.subplot(2, 2, 1), plt.imshow(img), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(imgLap, cmap='gray'), plt.title('Laplacian Edge'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(imgOtsu, cmap='gray'), plt.title('Otsu Method'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(imgAdapt, cmap='gray'), plt.title('Adaptive Gaussian Threshold'), plt.xticks(
    []), plt.yticks([])
plt.show()

