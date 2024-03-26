import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#加载数据
dataset = pd.read_csv('load.csv',index_col=[0])
dataset = dataset.fillna(method='pad')
#print(dataset)
#忘了
dataset = np.array(dataset)

a = []
for item in dataset:
    for i in item:
        a.append(i)
dataset = pd.DataFrame(a)
#print(dataset)
#print(a)

test = dataset.iloc[int(len(a)*0.98):,[0]]
#数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(test)

x_test = []
y_test = []

for i in np.arange(96,len(train)):
    x_test.append(train[i-96:i, :])
    y_test.append(train[i])
#print(x_test)

x_test, y_test = np.array(x_test),np.array(y_test)

model = load_model('LSTM_model.h5')
#test the model
predicted = model.predict(x_test)
#print(predicted)

#反归一化
prediction = scaler.inverse_transform(predicted)
#print(prediction)

real = scaler.inverse_transform(y_test)
print(real)

#画图对比
# #绘制真实值预测值对比图
# #创建（12，8）画布
# plt.figure(figsize=(12, 8))
# #传入真实值预测值
# plt.plot(prediction,label='预测值')
# plt.plot(real,label='真实值')
# #设置xy轴刻度值大小
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
#
# #设置xy轴标签
# plt.legend(loc='best',fontsize=15)
# plt.ylabel('负荷值', fontsize=15)
# plt.xlabel('采样点', fontsize=15)
# #设置图参数名字
# plt.title("基于LSTM神经网络负荷预测", fontsize=15)
# plt.show()

#模型评估
#调用模型评价指标
#R2
from sklearn.metrics import r2_score
#MSE
from sklearn.metrics import mean_squared_error
#MAE
from sklearn.metrics import mean_absolute_error

R2= r2_score(real,prediction)
MAE = mean_absolute_error(real, prediction)
RMSE = np.sqrt(mean_squared_error(real, prediction))
MAPE= np.mean(np.abs(real-prediction)/prediction)

print("R2: ", R2)
print("MAE: ", MAE)
print("RMSE: ", RMSE)
print("MAPE: ", MAPE)