import matplotlib
import matplotlib.pyplot as plt
import random as random
import numpy as np
import csv

# 假设x_data和y_data都有10笔，分别代表宝可梦进化前后的cp值
x_data=[338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data=[640.,633.,619.,393.,428.,27.,193.,66.,226.,1591.]
# 这里采用最简单的linear model：y_data=b+w*x_data
# 我们要用gradient descent把b和w找出来

# 计算梯度微分的函数getGrad()
def getGrad(b,w):
    # initial b_grad and w_grad
    b_grad = 0.0
    w_grad = 0.0
    # 计算梯度
    for i in range(len(x_data)):
        b_grad += -2 * (y_data[i] - b - w * x_data[i])
        w_grad += -2 * x_data[i] * (y_data[i] - b - w * x_data[i])
    return b_grad, w_grad


# y_data = b + w * x_data
b = -120 # initial b
w = -4 # initial w
lr = 0.00000114 # learning rate
iteration = 5000000 # 这里直接规定了迭代次数，而不是一直运行到b_grad和w_grad都为0
for i in range(iteration):
    b_grad, w_grad = getGrad(b,w)
    b -= lr * b_grad
    w -= lr * w_grad
print(w,b)

