# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/11/14 21:43
import cv2.cv2 as cv2
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt


class ImageContraster:
    def __init__(self):
        pass

    def enhance_contrast(self, img, method="HE", level=256, window_size=32, affect_size=16, blocks=8, threshold=10.0):
        if method in ['HE', 'FHE', 'he', 'fhe']:
            he_func = self.histogram_equalization
        elif method in ['Bright', 'bright', 'bright_level']:
            he_func = self.bright_wise_histequal

        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            channel_num = 1
        elif len(img_arr.shape) == 3:
            channel_num = img_arr.shape[2]

        if channel_num == 1:
            # gray image
            arr = he_func(img_arr, level=level, window_size=window_size, affect_size=affect_size, blocks=blocks,
                          threshold=threshold)
            img_res = Image.fromarray(arr)
        elif channel_num == 3 or channel_num == 4:
            # RGB image or RGBA image(such as png)
            rgb_arr = [None] * 3
            rgb_img = [None] * 3
            # process dividely
            for k in range(3):
                rgb_arr[k] = he_func(img_arr[:, :, k], level=level, window_size=window_size, affect_size=affect_size,
                                     blocks=blocks, threshold=threshold)
                rgb_img[k] = Image.fromarray(rgb_arr[k])
            img_res = Image.merge("RGB", tuple(rgb_img))
            # img_res = np.array(img_res)
            # cv2.imwrite('C:/Users/sycui/Desktop/Bright.jpg', img_res)
        return img_res

    def histogram_equalization(self, img_arr, level=256, **args):

        hists = self.calc_histogram_(img_arr, level)
        (m, n) = img_arr.shape
        hists_cdf = self.calc_histogram_cdf_(hists, m, n, level)  # calculate CDF
        arr = np.zeros_like(img_arr)
        arr = hists_cdf[img_arr]  # mapping
        return arr

    def bright_wise_histequal(self, img_arr, level=256, **args):
        def special_histogram(img_arr, min_v, max_v):

            hists = [0 for _ in range(max_v - min_v + 1)]
            for v in img_arr:
                hists[v - min_v] += 1
            return hists

        def special_histogram_cdf(hists, min_v, max_v):
            hists_cumsum = np.cumsum(np.array(hists))
            hists_cdf = (max_v - min_v) / hists_cumsum[-1] * hists_cumsum + min_v
            hists_cdf = hists_cdf.astype("uint8")
            return hists_cdf

        def pseudo_variance(arr):
            arr_abs = np.abs(arr - np.mean(arr))
            return np.mean(arr_abs)

        (m, n) = img_arr.shape
        hists = self.calc_histogram_(img_arr)
        hists_arr = np.cumsum(np.array(hists))
        hists_ratio = hists_arr / hists_arr[-1]

        scale1 = None
        scale2 = None
        for i in range(len(hists_ratio)):
            if hists_ratio[i] >= 0.333 and scale1 == None:
                scale1 = i
            if hists_ratio[i] >= 0.667 and scale2 == None:
                scale2 = i
                break

        # split images
        dark_index = (img_arr <= scale1)
        mid_index = (img_arr > scale1) & (img_arr <= scale2)
        bright_index = (img_arr > scale2)

        # variance
        dark_variance = pseudo_variance(img_arr[dark_index])
        mid_variance = pseudo_variance(img_arr[mid_index])
        bright_variance = pseudo_variance(img_arr[bright_index])

        # build three level images
        dark_img_arr = np.zeros_like(img_arr)
        mid_img_arr = np.zeros_like(img_arr)
        bright_img_arr = np.zeros_like(img_arr)

        # histogram equalization individually
        dark_hists = special_histogram(img_arr[dark_index], 0, scale1)
        dark_cdf = special_histogram_cdf(dark_hists, 0, scale1)

        mid_hists = special_histogram(img_arr[mid_index], scale1, scale2)
        mid_cdf = special_histogram_cdf(mid_hists, scale1, scale2)

        bright_hists = special_histogram(img_arr[bright_index], scale2, level - 1)
        bright_cdf = special_histogram_cdf(bright_hists, scale2, level - 1)

        def plot_hists(arr):
            hists = [0 for i in range(256)]
            for a in arr:
                hists[a] += 1
            self.draw_histogram_(hists)

        # mapping
        dark_img_arr[dark_index] = dark_cdf[img_arr[dark_index]]
        mid_img_arr[mid_index] = mid_cdf[img_arr[mid_index] - scale1]
        bright_img_arr[bright_index] = bright_cdf[img_arr[bright_index] - scale2]

        arr = dark_img_arr + mid_img_arr + bright_img_arr
        arr = arr.astype("uint8")
        return arr

    def standard_histogram_equalization(self, img_arr, level=256, **args):
        # ImageOps.equalize
        img = Image.fromarray(img_arr)
        img_res = ImageOps.equalize(img)
        arr = np.array(img_res)
        return arr

    def calc_histogram_(self, gray_arr, level=256):
        hists = [0 for _ in range(level)]
        for row in gray_arr:
            for p in row:
                hists[p] += 1
        return hists

    def calc_histogram_cdf_(self, hists, block_m, block_n, level=256):
        hists_cumsum = np.cumsum(np.array(hists))
        const_a = (level - 1) / (block_m * block_n)
        hists_cdf = (const_a * hists_cumsum).astype("uint8")
        return hists_cdf

    def draw_histogram_(self, hists):
        plt.figure()
        plt.bar(range(len(hists)), hists)
        plt.show()

    def plot_images(self, img1, img2):
        plt.figure()
        plt.subplot(121)
        plt.imshow(img1)
        plt.subplot(122)
        plt.imshow(img2)
        plt.show()
