# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/10/26 14:49
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist


def equalHist(img):
    # 灰度图像矩阵的高、宽
    h, w = img.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(img)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]
    return equalHistImage


img = cv2.imread('D:/Mobilenet_CoordinateAttention_GAF/8.jpg')
print(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
dst = clahe.apply(img)
equa = cv2.equalizeHist(img)


# cv2.imshow('img', img)
# cv2.imshow('dit', dst)
# cv2.imshow('equa', equa)
# cv2.waitKey()
blur1 = cv2.GaussianBlur(equa, (5, 5), sigmaX=0)  # 用高斯滤波处理原图像降噪
canny1 = cv2.Canny(blur1, 50, 120)  # 50是最小阈值,150是最大阈值

blur2 = cv2.medianBlur(equa, 9)  # 用高斯滤波处理原图像降噪
canny2 = cv2.Canny(blur2, 50, 120)  # 50是最小阈值,150是最大阈值

rho = 1  # 距离分辨率
theta = np.pi / 180  # 角度分辨率
threshold = 10  # 霍夫空间中多少个曲线相交才算作正式交点
min_line_len = 100  # 最少多少个像素点才构成一条直线
max_line_gap = 150  # 线段之间的最大间隔像素
lines = cv2.HoughLinesP(canny2, rho, theta, threshold, maxLineGap=max_line_gap)
line_img = np.zeros_like(canny2)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 1)
cv2.imshow("line_img", line_img)
cv2.waitKey()


plt.figure(1)
# 第一行第一列图形
ax1 = plt.subplot(1, 5, 1)
plt.sca(ax1)
plt.imshow(equa)
plt.title("artwork")

ax2 = plt.subplot(1, 5, 2)
plt.sca(ax2)
plt.imshow(blur1, cmap="gray")
plt.title("Gaussian")

# 第一行第二列图形
ax3 = plt.subplot(1, 5, 3)
plt.sca(ax3)
plt.imshow(canny1, cmap="gray")
plt.title("Gaussian Canny")

ax4 = plt.subplot(1, 5, 4)
plt.sca(ax4)
plt.imshow(blur2, cmap="gray")
plt.title("Median")

ax5 = plt.subplot(1, 5, 5)
plt.sca(ax5)
plt.imshow(canny2, cmap="gray")
plt.title("Median Canny")
plt.show()
