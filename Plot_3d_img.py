# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/5/26 22:09
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from math import *
from mpl_toolkits.mplot3d import Axes3D
import csv
import codecs


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        print(data)
        writer.writerow(data)
    print("保存文件成功，处理结束")


def plt3ddraw(dir):
    dt = pd.read_csv(dir)
    x = dt['X118_DE_time']
    x_ = (x - max(x) + x - min(x)) / (max(x) - min(x))

    fig = plt.figure()
    ax = Axes3D(fig)
    X = x_[:50]

    Y = x_[:50]

    X, Y = np.meshgrid(X, Y)
    Z = X * Y
    # ax.grid(False)
    # ax.set_xticks([])  # 不显示x坐标轴
    # ax.set_yticks([])  # 不显示y坐标轴
    # ax.set_zticks([])  # 不显示z坐标轴
    # plt.axis('off')#关闭所有坐标轴
    plt.subplots_adjust(0, 0, 1, 1)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()


if __name__ == "__main__":
    path = r'D:\pythonProject2\GAF-CNN(s)\data_set_5000.csv'
    plt3ddraw(path)
