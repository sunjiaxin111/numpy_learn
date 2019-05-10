import numpy as np

# 重点：NumPy数组就是ndarray对象的嵌套
# NumPy这个词来源于两个单词--Numerical和Python。

# NumPy提供的最重要的数据结构是一个称为NumPy数组的强大对象。
my_array = np.array([1, 2, 3, 4, 5])  # 创建一个NumPy数组
print(my_array)  # 打印NumPy数组
print(my_array.shape)  # 打印NumPy数组的形状
print(my_array[0])  # 打印NumPy数组的元素
print(my_array[1])
my_array[0] = -1  # 修改NumPy数组的元素
print(my_array)
my_new_array = np.zeros((5))  # 创建一个长度为5的NumPy数组，但所有元素都为0
print(my_new_array)
my_new_array1 = np.ones((5))
print(my_new_array1)
my_random_array = np.random.random((5))  # 创建一个随机值数组
print(my_random_array)
my_2d_array = np.zeros((2, 3))  # 创建二维数组
print(my_2d_array)
my_2d_array_new = np.ones((2, 4))
print(my_2d_array_new)
my_array = np.array([[4, 5], [6, 1]])
print(my_array[0][1])  # 按照索引取值
print(my_array.shape)
my_array_column_2 = my_array[:, 1]  # 提取第二列（索引1）的所有元素
print(my_array_column_2)

# NumPy中的数组操作
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])
sum = a + b
difference = a - b
product = a * b
quotient = a / b
print("Sum = \n", sum)
print("Difference = \n", difference)
print("Product = \n", product)
print("Quotient = \n", quotient)

# 执行矩阵乘法
matrix_product = a.dot(b)
print("Matrix Product = \n", matrix_product)
