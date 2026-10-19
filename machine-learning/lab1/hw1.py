####补全代码，添加注释 2024/11/5 due
import sys
import pandas as pd
import numpy as np
data = pd.read_csv('./train.csv') #train里面是12个月每个月的前20天

data = data.iloc[:, 3:] #从数据开始
data = data.replace('NR', 0) #把NR数据化，不然算不了
data = data.astype(float)
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480]) #18行，480列，行是测项，列是20天x24小时，就是20天里面的每个小时
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :] #一天18行，一个月20天
    month_data[month] = sample
    
 
x = np.empty([12 * 471, 18 * 9], dtype = float) #滑动窗口，每次滑动9小时，滑倒471
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) 
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] 
print(x)
print(y)

mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 ，标准差
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j] #标准化
            
import math
x_train_set = x[: math.floor(len(x) * 0.8), :]  #前80%训练集，后20%验证集
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
print(x_train_set)
print(y_train_set)
print(x_validation)
print(y_validation)
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))  

dim = 18 * 9 + 1 # y = b + w1x1 + w2x2 + ... + w_162 x_162，可以把b看成w0,y=np.dot(w,x)
w = np.zeros([dim,1])
x = np.concatenate((np.ones([len(x_train_set), 1]), x_train_set), axis=1).astype(float)  #b = w0x0
learning_rate = 10
iter_time = 100000
adagrad = np.zeros([dim, 1])
eps = 1e-10 # epsilon
for t in range(iter_time):
    loss = np.mean((np.dot(x,w)-y_train_set)**2)
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    gradient = 2*np.dot(x.T,np.dot(x,w)-y_train_set) / len(x_train_set) #MSE求导
    adagrad = adagrad + gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)

         
testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
## test_data[test_data == 'NR'] = 0
test_data = test_data.replace('NR',0)
test_data = test_data.astype(float)
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

# 读取训练好的权重
w = np.load('weight.npy')

# 预测
ans_y = test_x @ w

# 保存成 csv
result = pd.DataFrame({
    'id': [f'id_{i}' for i in range(240)],
    'value': ans_y.reshape(-1)
})

result.to_csv('result.csv', index=False)