# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.utils import np_utils,plot_model
from sklearn.model_selection import cross_val_score,train_test_split
from keras.layers import Dense, Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras import backend as K
 
# 载入数据
df = pd.read_csv(r"C:\Users\Desktop\数据集-用做回归.csv")
X = np.expand_dims(df.values[:, 0:246].astype(float), axis=2)#增加一维轴
Y = df.values[:, 246]
 
# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
 
# 自定义度量函数
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
 
# 定义一个神经网络
model = Sequential()
model.add(Conv1D(16, 3,input_shape=(246,1), activation='relu'))
model.add(Conv1D(16, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
plot_model(model, to_file='./model_linear.png', show_shapes=True)
print(model.summary())
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
 
# 训练模型
model.fit(X_train,Y_train, validation_data=(X_test, Y_test),epochs=40, batch_size=10)
 
# # 将其模型转换为json
# model_json = model.to_json()
# with open(r"C:\Users\Desktop\model.json",'w')as json_file:
#     json_file.write(model_json)# 权重不在json中,只保存网络结构
# model.save_weights('model.h5')
#
# # 加载模型用做预测
# json_file = open(r"C:\Users\Desktop\model.json", "r")
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model.h5")
# print("loaded model from disk")
# scores = model.evaluate(X_test,Y_test,verbose=0)
# print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
 
# 准确率
scores = model.evaluate(X_test,Y_test,verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
 
# 预测值散点图
predicted = model.predict(X_test)
plt.scatter(Y_test,predicted)
x=np.linspace(0,0.3,100)
y=x
plt.plot(x,y,color='red',linewidth=1.0,linestyle='--',label='line')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.legend(["y = x","湿度预测值"])
plt.title("预测值与真实值的偏离程度")
plt.xlabel('真实湿度值')
plt.ylabel('湿度预测值')
plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
plt.show()
 
# 计算误差
result =abs(np.mean(predicted - Y_test))
print("The mean error of linear regression:")
print(result)