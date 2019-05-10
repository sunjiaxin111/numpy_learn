import numpy as np

# 数组基础
# 使用NumPy表示向量（一维数组）,下面是创建数组的4种不同方法
a = np.array([0, 1, 2, 3, 4])  # 传递列表
b = np.array((0, 1, 2, 3, 4))  # 传递元组
c = np.arange(0, 5, 1)  # 调用np.arange方法
d = np.linspace(0, 2 * np.pi, 5)  # 调用np.linspace方法，等距

print(a)  # 输出[0 1 2 3 4]
print(b)  # 输出[0 1 2 3 4]
print(c)  # 输出[0 1 2 3 4]
print(d)  # 输出[0.         1.57079633 3.14159265 4.71238898 6.28318531]
print(a[3])  # 输出3

# 使用多维数组表示矩阵
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
print(a)
print(a[2][4])
print(a[2, 4])

# 多维数组切片
print(a[0, 1:4])  # 输出[12 13 14]
print(a[1:4, 0])  # 输出[16 21 26]
print(a[::2, ::2])
'''
输出
[[11 13 15]
 [21 23 25]
 [31 33 35]]
'''
print(a[:, 1])  # 输出[12 17 22 27 32]

# 数组属性
print(type(a))  # 输出<class 'numpy.ndarray'>
print(a.dtype)  # 输出int32
print(a.size)  # 输出25
print(a.shape)  # 输出(5, 5)
print(a.itemsize)  # 输出4，itemsize属性是每个项占用的字节数
print(a.ndim)  # 输出2，ndim 属性是数组的维数
print(a.nbytes)  # 输出100，nbytes 属性是数组中的所有数据消耗掉的字节数

# 使用数组
# 基本操作符
a = np.arange(25)
a = a.reshape((5, 5))
b = np.arange(25)
b = b.reshape((5, 5))
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)
print(a < b)
print(a > b)

# dot函数
# 处理一维数组时，相当于内积
c = np.array([1, 2])
d = np.array([1, 2])
print(c.dot(d))
# 处理二维数组时，为矩阵乘法
print(a.dot(b))
# 只要有一个是多维数组，则为矩阵乘法
e = np.array([1, 2, 3, 4, 5])
print(a.dot(e))
print(e.dot(a))

# 数组特殊运算符
a = np.arange(10)
print(a.sum())
print(a.min())
print(a.max())
print(a.cumsum())  # 输出[ 0  1  3  6 10 15 21 28 36 45]

# 花式索引
a = np.arange(0, 100, 10)
indices = [1, 5, -1]
b = a[indices]
print(a)  # 输出[ 0 10 20 30 40 50 60 70 80 90]
print(b)  # 输出[10 50 90]

# 布尔屏蔽
import matplotlib.pyplot as plt

a = np.linspace(0, 2 * np.pi, 50)
b = np.sin(a)
plt.plot(a, b)
mask = b >= 0
plt.plot(a[mask], b[mask], 'bo')
mask = (b >= 0) & (a <= np.pi / 2)
plt.plot(a[mask], b[mask], 'go')
plt.show()

# 缺省索引
a = np.arange(0, 100, 10)
b = a[:5]
c = a[a >= 50]
print(b)  # 输出[ 0 10 20 30 40]
print(c)  # 输出[50 60 70 80 90]

# Where函数,返回条件为真的索引列表
a = np.arange(0, 100, 10)
b = np.where(a < 50)
c = np.where(a >= 50)[0]
print(b)  # 输出(array([0, 1, 2, 3, 4], dtype=int64),)
print(c)  # 输出[5 6 7 8 9]
print(a[c])
