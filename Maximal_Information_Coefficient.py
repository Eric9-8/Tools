# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/4/18 22:29
import numpy as np
from minepy import MINE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 从硬盘读取数据进入内存
wine = pd.read_csv(r"C:\Users\sycui\Desktop\10.6数学建模\问题2\原始数据集\MIC分析使用2月.csv",encoding='gb2312')


# (‘位置/命名.格式’)格式在保存数据文档时选择的

def MIC_matirx(dataframe, mine):
    data = np.array(dataframe)
    n = len(data[0, :])
    result = np.zeros([n, n])

    for i in range(n):
        for j in range(n):
            mine.compute_score(data[:, i], data[:, j])
            result[i, j] = mine.mic()
            result[j, i] = mine.mic()
    RT = pd.DataFrame(result)
    return RT


mine = MINE(alpha=0.6, c=15)
data_wine_mic = MIC_matirx(wine, mine)
print(data_wine_mic)  # 输出结果data_wine_mic


#
# # 把数值结果写入到excel中(‘excel命名.格式’,想要写入的数据结果,分隔符逗号结束)
# np.savetxt('resultc.csv', data_wine_mic, delimiter=',')

# data_wine_mic.to_csv('./time_fre_mic.csv', sep=',')


# 可视化处理（热力图展示整体结果）
def ShowHeatMap(DataFrame):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(30, 40))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(DataFrame.astype(float), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',
                annot=True)
    plt.savefig(r'C:\Users\sycui\Desktop\10.6数学建模\问题2\原始数据集\mic2月.jpg')
    plt.show()


ShowHeatMap(data_wine_mic)
