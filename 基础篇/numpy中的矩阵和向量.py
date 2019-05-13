import csv

import numpy as np

# 构造一个矩阵
A = np.array([[1, -1, 2], [3, 2, 0]])
print(A)
# 构造一个向量，向量只是具有单列的数组
v = np.array([[2], [1], [3]])
print(v)
# 也可以通过转置行向量来得到列向量
v = np.transpose(np.array([[2, 1, 3]]))
print(v)
# 一维数组的转置等于其本身
print(np.transpose(np.array([2, 1, 3])).shape)

# numpy重载数组索引和切片符号以访问矩阵的各个部分
print(A[1, 2])
print(A[:, 1:2])

# 要进行矩阵乘法或矩阵向量乘法，我们使用np.dot()方法。
w = np.dot(A, v)
print(w)

# 用numpy求解方程组
# 求解Ax = b
# 构建A和b的数组
A = np.array([[2, 1, -2], [3, 0, 1], [1, 1, -1]])
b = np.transpose(np.array([[-3, 5, -2]]))
print(A)
print(b)
# 直接调用函数求解
x = np.linalg.solve(A, b)
print(x)


# 应用：多元线性回归
def readData():
    X = []
    y = []
    with open('Housing.csv') as f:
        rdr = csv.reader(f)
        # Skip the header row
        next(rdr)
        # Read X and y
        for line in rdr:
            xline = [1.0]
            for s in line[:-1]:
                xline.append(float(s))
            X.append(xline)
            y.append(float(line[-1]))
    return (X, y)


X0, y0 = readData()
# Convert all but the last 10 rows of the raw data to numpy arrays
d = len(X0) - 10
X = np.array(X0[:d])
y = np.transpose(np.array([y0[:d]]))

# Compute beta
Xt = np.transpose(X)
XtX = np.dot(Xt, X)
Xty = np.dot(Xt, y)
beta = np.linalg.solve(XtX, Xty)
print(beta)

# Make predictions for the last 10 rows in the data set
for data, actual in zip(X0[d:], y0[d:]):
    x = np.array([data])
    prediction = np.dot(x, beta)
    print('prediction = ' + str(prediction[0, 0]) + ' actual = ' + str(actual))
