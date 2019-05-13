import numpy as np

'''
创建Numpy数组有三种不同的方法：
1、使用Numpy内部功能函数
2、从列表等其他Python的结构进行转换
3、使用特殊的库函数
'''

# 使用Numpy内部功能函数
array = np.arange(20)
print(array)
print(array.shape)
print(array[3])
# Numpy数组是可变的
array[3] = 100
print(array)
# 与Python列表不同，Numpy数组的内容是同质的
# array[3] = 'Numpy'  # ValueError: invalid literal for int() with base 10: 'Numpy'

array = np.arange(20).reshape(4, 5)
print(array)
print(array.shape)
print(array[3][4])

array = np.arange(27).reshape(3, 3, 3)
print(array)
print(array.shape)

# 使用arange函数，你可以创建一个在定义的起始值和结束值之间具有特定序列的数组。
print(np.arange(10, 35, 3))

print(np.zeros((2, 4)))
print(np.ones((3, 4)))
print(np.empty((2, 3)))  # 它的初始内容是随机的，取决于内存的状态。
print(np.full((2, 2), 3))  # full函数创建一个填充给定值的n * n数组。
print(np.eye(3, 3))  # eye函数可以创建一个n * n矩阵，对角线为1s，其他为0。
print(np.eye(4, 3))
print(np.linspace(0, 10, num=4))  # 函数linspace在指定的时间间隔内返回均匀间隔的数字。

# 从Python列表转换
array = np.array([4, 5, 6])
print(array)

list = [4, 5, 6]
print(list)

array = np.array(list)
print(array)

# 改变list的值,不影响Numpy数组的值
list[0] = -1
print(list)
print(array)

print(type(list))
print(type(array))

array = np.array([(1, 2, 3), (4, 5, 6)])
print(array)
print(array.shape)

# 使用特殊的库函数
print(np.random.random((2, 2)))
