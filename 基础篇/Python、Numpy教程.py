'''
Python允许你在非常少的代码行中表达非常强大的想法，
同时非常具有可读性。下面是Python中经典快速排序算法的实现
'''


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    key = arr[len(arr) // 2]  # 取数组中间的值
    left = [x for x in arr if x < key]  # 小于key的值
    middle = [x for x in arr if x == key]  # 等于key的值
    right = [x for x in arr if x > key]  # 大于key的值

    return quicksort(left) + middle + quicksort(right)


print(quicksort([3, 6, 8, 10, 1, 2, 1]))

'''
基本数据类型
与大多数语言一样，Python有许多基本类型，
包括整数，浮点数，布尔值和字符串。
这些数据类型的行为方式与其他编程语言相似。
'''

# Numbers(数字类型):代表的是整数和浮点数
# Python中没有x++和x--运算符
x = 3
print(type(x))  # Prints "<class 'int'>"
print(x)  # Prints "3"
print(x + 1)  # Addition; prints "4"
print(x - 1)  # Subtraction; prints "2"
print(x * 2)  # Multiplication; prints "6"
print(x ** 2)  # Exponentiation; prints "9"
x += 1
print(x)  # Prints "4"
x *= 2
print(x)  # Prints "8"
y = 2.5
print(type(y))  # Prints "<class 'float'>"
print(y, y + 1, y * 2, y ** 2)  # Prints "2.5 3.5 5.0 6.25"

# Booleans(布尔类型)
t = True
f = False
print(type(t))  # Prints "<class 'bool'>"
print(t and f)  # Logical AND; prints "False"
print(t or f)  # Logical OR; prints "True"
print(not t)  # Logical NOT; prints "False"
print(t != f)  # Logical XOR; prints "True"

# Strings(字符串类型) %d表示的是十进制整数decimal
hello = 'hello'  # String literals can use single quotes
world = "world"  # or double quotes; it does not matter.
print(hello)  # Prints "hello"
print(len(hello))  # String length; prints "5"
hw = hello + ' ' + world  # String concatenation
print(hw)  # prints "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"

# String对象有许多有用的方法
s = "hello"
print(s.capitalize())  # 将字符串的第一个字母变成大写,其他字母变小写; prints "Hello"
print(s.upper())  # 将字符串的所有字母变成大写; prints "HELLO"
print(s.rjust(7))  # 返回一个原字符串右对齐,并使用空格填充至长度 width 的新字符串; prints "  hello"
print(s.center(7))  # 返回一个原字符串居中,并使用空格填充至长度 width 的新字符串; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
# prints "he(ell)(ell)o"
print('  world '.strip())  # 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列; prints "world"

'''
容器(Containers)
Python包含几种内置的容器类型：列表、字典、集合和元组
'''

# 列表(Lists)
# 列表其实就是Python中的数组，但是可以它可以动态的调整大小并且可以包含不同类型的元素
xs = [3, 1, 2]
print(xs, xs[2])  # 输出[3, 1, 2] 2
print(xs[-1])  # 输出2
xs[2] = 'foo'
print(xs)  # 输出[3, 1, 'foo']
xs.append('bar')  # 在列表尾部添加一个新元素
print(xs)  # 输出[3, 1, 'foo', 'bar']
x = xs.pop()  # 删除并返回列表中的最后一个元素
print(x, xs)  # 输出bar [3, 1, 'foo']

# 切片(Slicing): 除了一次访问一个列表元素之外，
# Python还提供了访问子列表的简明语法; 这被称为切片
nums = list(range(5))
print(nums)  # 输出[0, 1, 2, 3, 4]
print(nums[2:4])  # 输出[2, 3]
print(nums[2:])  # 输出[2, 3, 4]
print(nums[:2])  # 输出[0, 1]
print(nums[:])  # 输出[0, 1, 2, 3, 4]
print(nums[:-1])  # 输出[0, 1, 2, 3]
nums[2:4] = [8, 9]  # 给切片赋值
print(nums)  # 输出[0, 1, 8, 9, 4]

# (循环)Loops: 你可以循环遍历列表的元素
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)

# 如果要访问循环体内每个元素的索引，请使用内置的 enumerate 函数
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

# 列表推导式(List comprehensions)
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)  # Prints [0, 1, 4, 9, 16]

nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"

# 字典存储（键，值）对
d = {'cat': 'cute', 'dog': 'furry'}
print(d['cat'])
print('cat' in d)
d['fish'] = 'wet'
print(d['fish'])
print(d.get('monkey', 'N/A'))
print(d.get('fish', 'N/A'))
del d['fish']
print(d.get('fish', 'N/A'))

# (循环)Loops:默认是迭代字典中的键
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))

# 使用items方法来访问键及其对应的值
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))
print(d)

# 字典推导式(Dictionary comprehensions)
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # Prints "{0: 0, 2: 4, 4: 16}"

# 集合(Sets):集合是不同元素的无序集合
animals = {'cat', 'dog'}
print('cat' in animals)  # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
animals.add('fish')  # Add an element to a set
print('fish' in animals)  # Prints "True"
print(len(animals))  # Number of elements in a set; prints "3"
animals.add('cat')  # Adding an element that is already in the set does nothing
print(len(animals))  # Prints "3"
animals.remove('cat')  # Remove an element from a set
print(len(animals))  # Prints "2"

# 循环(Loops): 无序
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

# 集合推导式(Set comprehensions)
from math import sqrt

nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # Prints "{0, 1, 2, 3, 4, 5}"

'''
元组(Tuples)
元组是（不可变的）有序值列表。 
元组在很多方面类似于列表; 
其中一个最重要的区别是元组可以用作字典中的键和集合的元素，
而列表则不能。
'''
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)  # Create a tuple
print(type(t))  # Prints "<class 'tuple'>"
print(d[t])  # Prints "5"
print(d[(1, 2)])  # Prints "1"


# 函数(Functions)
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'


for x in [-1, 0, 1]:
    print(sign(x))


def hello(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)


hello('Bob')  # Prints "Hello, Bob"
hello('Fred', loud=True)  # Prints "HELLO, FRED!"


# 类(Classes)
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)


g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()  # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)  # Call an instance method; prints "HELLO, FRED!"

# Numpy
import numpy as np

a = np.array([1, 2, 3])  # Create a rank 1 array
print(type(a))  # Prints "<class 'numpy.ndarray'>"
print(a.shape)  # Prints "(3,)"
print(a[0], a[1], a[2])  # Prints "1 2 3"
a[0] = 5  # Change an element of the array
print(a)  # Prints "[5, 2, 3]"

b = np.array([[1, 2, 3], [4, 5, 6]])  # Create a rank 2 array
print(b.shape)  # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])  # Prints "1 2 4"

a = np.zeros((2, 2))  # 创建一个全为0的数组
print(a)

b = np.ones((1, 2))  # 创建一个全为1的数组
print(b)

c = np.full((2, 2), 7)  # 创建一个常量数组
print(c)

d = np.eye(2)  # 创建一个单位矩阵
print(d)

e = np.random.random((2, 2))  # 创建一个随机数数组
print(e)

# 数组索引
# 切片
# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# 使用切片索引到numpy数组时，生成的数组视图将始终是原始数组的子数组。
# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])  # Prints "2"
b[0, 0] = 77  # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])  # Prints "77"

# 将整数索引与切片索引混合使用
# 但是，这样做会产生比原始数组更低级别的数组。
# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]  # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)

col_r1[0] = -1
print(a)

# 整数数组索引
# 整数数组索引允许你使用另一个数组中的数据构造任意数组,
# 构造出来的数组中的数据与原始数组无关
a = np.array([[1, 2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"

b = np.array([a[0, 0], a[1, 1], a[2, 0]])
b[0] = -1
print(a)
print(b)

b = a[[0, 1, 2], [0, 1, 0]]
b[0] = -1
print(a)
print(b)

# 整数数组索引的一个有用技巧是从矩阵的每一行中选择或改变一个元素
# Create a new array from which we will select elements
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

print(a)

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)

# 布尔数组索引: 布尔数组索引允许你选择数组的任意元素。
# 通常，这种类型的索引用于选择满足某些条件的数组元素。
a = np.array([[1, 2], [3, 4], [5, 6]])

bool_idx = (a > 2)
print(bool_idx)

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])  # Prints "[3 4 5 6]"

# 在构造Numpy数组时可以显式指定数据类型
x = np.array([1, 2])  # Let numpy choose the datatype
print(x.dtype)  # Prints "int64"

x = np.array([1.0, 2.0])  # Let numpy choose the datatype
print(x.dtype)  # Prints "float64"

x = np.array([1, 2], dtype=np.int64)  # Force a particular datatype
print(x.dtype)  # Prints "int64"

# 数组中的数学
# 基本数学函数在数组上以元素方式运行，
# 既可以作为运算符重载，也可以作为numpy模块中的函数
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))

print(x / y)
print(np.divide(x, y))

print(np.sqrt(x))

# *是元素乘法，dot函数为向量内积或矩阵乘法
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

v = np.array([9, 10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))

# SUM函数
x = np.array([[1, 2], [3, 4]])

# SUM函数的axis参数表示按第几个维度相加
print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # 按第0个维度相加，[1,2]+[3,4]=[4,6]
print(np.sum(x, axis=1))  # 按第1个维度相加，[1,3]+[2,4]=[3,7]

# 转置
x = np.array([[1, 2], [3, 4]])
print(x)
print(x.T)

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1, 2, 3])
print(v)  # Prints "[1 2 3]"
print(v.T)  # Prints "[1 2 3]"

# 广播
# 广播是一种强大的机制，它允许numpy在执行算术运算时使用不同形状的数组
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)  # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))  # reps的数字从后往前分别对应A的第N个维度的重复次数
print(vv)
y = x + vv  # Add x and vv elementwise
print(y)

# Numpy广播允许我们在不实际创建v的多个副本的情况下执行此计算
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)

# 以下是广播的一些应用
# Compute outer product of vectors
v = np.array([1, 2, 3])  # v has shape (3,)
w = np.array([4, 5])  # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w)

# Add a vector to each row of a matrix
x = np.array([[1, 2, 3], [4, 5, 6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(x + v)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((x.T + w).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(x * 2)

# Scipy
# 图像操作
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')
print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)

# 点之间的距离
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print(d)

# Matplotlib
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.

# 通过一些额外的工作，我们可以轻松地一次绘制多条线，并添加标题，图例和轴标签
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])  # 图例
plt.show()

# 子图
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

# 图片
img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))  # 这里要转成uint8格式
plt.show()
