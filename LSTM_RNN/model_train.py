import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
import keras
import numpy as np

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

train = dataset.iloc[0: int(len(a)*0.8),[0]]
val = dataset.iloc[int(len(a)*0.8):int(len(a)*0.9),[0]]

#数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(train)
val = scaler.fit_transform(val)
#print(train)
#print(val)

x_train = []
y_train = []

for i in np.arange(96,len(train)):
    x_train.append(train[i-96:i, :])
    y_train.append(train[i])
#print(x_train)

x_train, y_train = np.array(x_train),np.array(y_train)
# print(x_train.shape)
# print(y_train.shape)

x_val = []
y_val = []

for i in np.arange(96,len(val)):
    x_val.append(val[i-96:i, :])
    y_val.append(val[i])
#print(x_val)

x_val, y_val = np.array(x_val),np.array(y_val)
# print(x_val.shape)
# print(y_val.shape)


#model
model = Sequential()
#model.add(SimpleRNN(10, return_sequences=True, activation='relu'))
model.add(LSTM(10, return_sequences=True, activation='relu'))
#model.add(SimpleRNN(15, return_sequences=False, activation='relu'))
model.add(LSTM(15, return_sequences=False, activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

#优化器(梯度下降的一种方法
model.compile(optimizer=keras.optimizers.Adam(0.01),loss='mse')

#网络训练
history = model.fit(x_train, y_train,batch_size=512,epochs=20,validation_data=(x_val,y_val))
#模型保存
model.save("LSTM_model.h5")

#绘制训练集测试集loss对比图
plt.figure(figsize=(12,8))
#传入训练集验证集loss
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')
#设置图参数名字
plt.title("LSTM神经网络loss值",fontsize=15)
#设置xy轴刻度值大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.ylabel('loss值',fontsize=15)
plt.xlabel('训练轮次',fontsize=15)

#设置图例文字大小
plt.legend(fontsize=15)
plt.show()