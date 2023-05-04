# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/11/20 22:47
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def histogram(grayfig):  # 绘制直方图
    x = grayfig.size[0]
    y = grayfig.size[1]
    ret = np.zeros(256)
    for i in range(x):  # 遍历像素点获得灰度值
        for j in range(y):
            k = grayfig.getpixel((i, j))
            ret[k] = ret[k] + 1
    for k in range(256):
        ret[k] = ret[k] / (x * y)
    return ret  # 返回包含各灰度值占比的数组


default = Image.open(r'C:\Users\sycui\Desktop\rail_33.jpg').convert('L')
gama = Image.open(r'C:\Users\sycui\Desktop\gmma_0.5.jpg').convert('L')
he = Image.open(r'C:\Users\sycui\Desktop\HE.jpg').convert('L')
adapt_he = Image.open(r'C:\Users\sycui\Desktop\Bright.jpg').convert('L')

fig = histogram(adapt_he)
plt.figure()
plt.bar(range(256), fig)
plt.show()