# 参考http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 计算损失，用了矢量化编程而不是for循环
def computeLoss(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# 梯度下降部分
def gradientDescent(X, y, theta, alpha, iters):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeLoss(X, y, theta)

    return theta, cost

# 读入训练数据
# windows用户路径可能需要修改下，后期有时间可能会做统一
def loadData(path):
    trainingData = pd.read_csv(path, header=None, names=['Population', 'Profit'])

    trainingData.head()

    trainingData.describe()

    trainingData.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
    plt.show()
    return trainingData

trainingData = loadData(os.getcwd() + '/../data/ex1data1.txt')

# 在数据集前插入一列Ones作为常数系数，也就是y=k*x+b*1这种形式
trainingData.insert(0, 'Ones', 1)

# 将输入X以及输出y从数据集中分割
cols = trainingData.shape[1]  
X = trainingData.iloc[:,0:cols-1]  
y = trainingData.iloc[:,cols-1:cols]  

# 把pandas的DataFrames转换成numpy的矩阵
X = np.matrix(X.values)  
y = np.matrix(y.values)  
# 初始化参数为全0的，当然也可以初始化成其他的
theta = np.matrix(np.array([0,0]))  

# 各向量的维度
X.shape, theta.shape, y.shape  

# 初始损失函数值
computeLoss(X, y, theta)   # 32.07，后面可以看看训练完后的损失函数值

# 设置学习速率以及迭代次数
alpha = 0.01  
iters = 2000

# 使用梯度下降得到模型参数
theta_fin, loss = gradientDescent(X, y, theta, alpha, iters)  
theta_fin

# 计算训练后的参数的损失值
computeLoss(X, y, theta_fin)  # 4.47

# 为了画线用的，画出训练好后的直线
x = np.linspace(trainingData.Population.min(), trainingData.Population.max(), 100)  
f = theta_fin[0, 0] + (theta_fin[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(trainingData.Population, trainingData.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size')  
plt.show()

# 损失随着迭代次数的变化
fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iters), loss, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Loss')  
ax.set_title('Error vs. Training Epoch')  
plt.show()