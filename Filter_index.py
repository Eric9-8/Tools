# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/11/8 22:16
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import imgvision as iv
import cv2.cv2 as cv2
import numpy as np

default = cv2.imread('C:/Users/sycui/Desktop/rail_33.jpg')
test = cv2.imread('C:/Users/sycui/Desktop/rail_33_noisy.jpg')
# 增加噪声
# 读取图片

# 设置高斯分布的均值和方差
# mean = 0
# # 设置高斯分布的标准差
# sigma = 10
# # 根据均值和标准差生成符合高斯分布的噪声
# img_height, img_width, img_channels = default.shape
# gauss = np.random.normal(mean, sigma, (img_height, img_width, img_channels))
# # 给图片添加高斯噪声
# noisy_img = default + gauss
# # 设置图片添加高斯噪声之后的像素值的范围
# noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
# # 保存图片
# cv2.imwrite('C:/Users/sycui/Desktop/rail_33_noisy.jpg', noisy_img)

# 均值滤波
out1 = cv2.blur(test, (5, 5))
cv2.imwrite('C:/Users/sycui/Desktop/mean.jpg', out1)
# 中值滤波
out2 = cv2.medianBlur(test, 5)
cv2.imwrite('C:/Users/sycui/Desktop/median.jpg', out2)


# 自适应均值滤波
def auto_median_filter(image, max_size):
    origen = 3  # 初始窗口大小
    board = origen // 2  # 初始应扩充的边界
    # max_board = max_size//2                                         # 最大可扩充的边界
    copy = cv2.copyMakeBorder(image, *[board] * 4, borderType=cv2.BORDER_DEFAULT)  # 扩充边界
    out_img = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            def sub_func(src, size):  # 两个层次的子函数
                kernel = src[i:i + size, j:j + size]
                # print(kernel)
                z_med = np.median(kernel)
                z_max = np.max(kernel)
                z_min = np.min(kernel)
                if z_min < z_med < z_max:  # 层次A
                    cc = image[i][j]
                    if z_min < all(image[i][j]) < z_max:  # 层次B
                        return image[i][j]
                    else:
                        return z_med
                else:
                    next_size = cv2.copyMakeBorder(src, *[1] * 4, borderType=cv2.BORDER_DEFAULT)  # 增尺寸
                    size = size + 2  # 奇数的核找中值才准确
                    if size <= max_size:
                        return sub_func(next_size, size)  # 重复层次A
                    else:
                        return z_med

            out_img[i][j] = sub_func(copy, origen)
    return out_img


out3 = auto_median_filter(test, 9)
cv2.imwrite('C:/Users/sycui/Desktop/auto_median.jpg', out3)

mean = cv2.imread('C:/Users/sycui/Desktop/mean.jpg')
median = cv2.imread('C:/Users/sycui/Desktop/median.jpg')
auto_median = cv2.imread('C:/Users/sycui/Desktop/auto_median.jpg')
Metrix = iv.spectra_metric(test, median, max_v=255)

# MSE
MSE = Metrix.MSE()
# PSNR
PSNR = Metrix.PSNR()
# SSIM
SSIM = Metrix.SSIM()
# ERGAS
ERGAS = Metrix.ERGAS()

print('================')
