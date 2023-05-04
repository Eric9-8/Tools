# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/10/26 14:10
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class Fileinfo:
    def __init__(self, path):
        self.path = path

    def getalldir(self):
        return os.listdir(self.path)

    def change_all_name(self, listdir):
        i = 0
        for file in listdir:
            if not os.path.isdir(file):
                i = i + 1
                os.rename(self.path + "\\" + file, self.path + "\\" + str(i) + file[-4:])


if __name__ == '__main__':
    fileinfo = Fileinfo(r'../WGAN_GP/dataset/损伤混合')  # 请忽视OpenCV
    fileinfo.change_all_name(fileinfo.getalldir())
