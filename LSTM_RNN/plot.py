#导入库 单特征用电负荷预测
#电工杯竞赛16年A题的数据集链接
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#加载历史数据文件
#index_col = [0], 将第一列作为索引
dataset = pd.read_csv('load.csv', index_col=[0])
dataset = dataset.fillna(method='pad')
#print(dataset)
dataset = np.array(dataset)
#print(dataset)
#将所有数据放到一个列表里，方便后续训练集测试集制作
a = []
for item in dataset:
    for i in item:
        a.append(i)
dataset = pd.DataFrame(a)

real = np.array(dataset)
#绘制真实值预测值对比图
#创建（12，8）画布
plt.figure(figsize=(20, 8))
#传入真实值预测值
plt.plot(real)
#设置xy轴刻度值大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
labels = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
plt.xticks(range(0, 35040, 2920), labels=labels)
#设置xy轴标签
plt.ylabel('负荷值', fontsize=15)
plt.xlabel('月份', fontsize=15)
#设置图参数名字
plt.title("数据总览", fontsize=15)
plt.show()

week_data = dataset.iloc[96*6:96*12,:]
c = np.array(week_data)
plt.figure(figsize=(20, 8))
#传入真实值预测值
plt.plot(week_data)
#设置xy轴刻度值大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
plt.xticks(range(0, 96*7, 96), labels=labels)
#设置xy轴标签
plt.ylabel('负荷值', fontsize=15)
plt.xlabel('日期', fontsize=15)
#设置图参数名字
plt.title("周数据", fontsize=15)
plt.show()