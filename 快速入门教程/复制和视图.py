import numpy as np

# 1、完全不复制
# 简单赋值不会创建数组对象或其数据的拷贝。
a = np.arange(12)
b = a
print(b is a)
b.shape = 3, 4
print(a.shape)


# Python将可变对象作为引用传递，所以函数调用不会复制。
def f(x):
    print(id(x))


print(id(a))  # id()是对象的唯一标识符
f(a)

# 2、视图或浅复制
# 不同的数组对象可以共享相同的数据。
# view 方法创建一个新的数组对象，它查看相同的数据。
c = a.view()
print(c is a)
print(c.base is a)
print(c.flags.owndata)
c.shape = 2, 6
print(a.shape)
c[0, 4] = 1234
print(a)

# 对数组切片返回一个视图
a[:, 1:3] = 10
print(a)

# 3、深拷贝
# copy 方法生成数组及其数据的完整拷贝。
d = a.copy()
print(d is a)
print(d.base is a)
d[0, 0] = 9999
print(a)

# 布尔索引
a = np.arange(12).reshape(3, 4)
b1 = np.array([False, True, True])
b2 = np.array([True, False, True, False])
print(a)
print(a[b1, b2])  # 输出[ 4 10]
