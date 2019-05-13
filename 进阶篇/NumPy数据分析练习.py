import numpy as np

# 问题：将numpy导入为 np 并打印版本号。
print(np.__version__)

# 问题：创建从0到9的一维数字数组
arr = np.arange(10)
print(arr)

# 问题：创建一个numpy数组元素值全为True（真）的数组
print(np.full((2, 2), True, dtype=bool))
print(np.ones((2, 2), dtype=bool))

# 问题：从 arr 中提取所有的奇数
print(arr[arr % 2 == 1])

# 问题：将arr中的所有奇数替换为-1。
arr = np.arange(10)
arr[arr % 2 == 1] = -1
print(arr)

# 问题：将arr中的所有奇数替换为-1，而不改变arr。
arr = np.arange(10)
# np.where如果只指定condition参数，则返回满足条件的索引
# 如果同时指定condition、x和y，则返回一个Numpy数组，在满足条件时用x，其他时用y
out = np.where(arr % 2 == 1, -1, arr)
print(out)
print(arr)

# 问题：将一维数组转换为2行的2维数组
print(np.arange(10).reshape(2, -1))

# 问题：垂直堆叠数组a和数组b
a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)
# 方法1
print(np.concatenate([a, b], axis=0))  # 增加第0个维度
# 方法2
print(np.vstack([a, b]))
# 方法3
print(np.r_[a, b])

# 问题：将数组a和数组b水平堆叠。
a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)
print(np.concatenate([a, b], axis=1))
print(np.hstack([a, b]))
print(np.c_[a, b])

# 问题：创建以下模式而不使用硬编码。只使用numpy函数和下面的输入数组a。
# 给定：
a = np.array([1, 2, 3])
# 期望的输出：
# > array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
print(np.r_[np.repeat(a, 3), np.tile(a, 3)])

# 问题：获取数组a和数组b之间的公共项。
# 给定：
a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
# 期望的输出：array([2, 4])
print(np.intersect1d(a, b))

# 问题：从数组a中删除数组b中的所有项。
# 给定：
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])
# 期望的输出：array([1,2,3,4])
print(np.setdiff1d(a, b))

# 问题：获取a和b元素匹配的位置。
# 给定：
a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
# 期望的输出：# > (array([1, 3, 5, 7]),)
print(np.where(a == b))

# 问题：获取5到10之间的所有项目。
# 给定：
a = np.array([2, 6, 1, 9, 10, 3, 27])
# 期望的输出：(array([6, 9, 10]),)
# 方法1
index = np.where((a >= 5) & (a <= 10))
print(a[index])
# 方法2
index = np.where(np.logical_and(a >= 5, a <= 10))
print(a[index])
# 方法3
print(a[(a >= 5) & (a <= 10)])


# 问题：转换适用于两个标量的函数maxx，以处理两个数组。
# 给定：
def maxx(x, y):
    """Get the maximum of two items"""
    if x >= y:
        return x
    else:
        return y


print(maxx(1, 5))


# > 5
# 期望的输出：
# a = np.array([5, 7, 9, 8, 6, 4, 5])
# b = np.array([6, 3, 4, 8, 9, 7, 1])
# pair_max(a, b)
# # > array([ 6.,  7.,  9.,  8.,  9.,  7.,  5.])
# 方法1
def pair_max(x, y):
    return np.where(x >= y, x, y)


a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
print(pair_max(a, b))

# 方法2:向量化函数
pair_max = np.vectorize(maxx, otypes=[float])
print(pair_max(a, b))

# 问题：在数组arr中交换列1和2。
arr = np.arange(9).reshape(3, 3)
print(arr[:, [1, 0, 2]])
print(arr)

# 问题：交换数组arr中的第1和第2行：
arr = np.arange(9).reshape(3, 3)
print(arr[[1, 0, 2], :])

# 问题：反转二维数组arr的行。
arr = np.arange(9).reshape(3, 3)
print(arr[::-1, :])

# 问题：反转二维数组arr的列。
print(arr[:, ::-1])

# 问题：创建一个形状为5x3的二维数组，以包含5到10之间的随机十进制数。
# 方法1：randint+random
# 其中numpy.random.random(size=None)生成[0,1)之间的浮点数
# numpy.random.randint(low, high=None, size=None, dtype='l')
print(np.random.randint(low=5, high=10, size=(5, 3)) + np.random.random((5, 3)))

# 方法2：numpy.random.uniform(low,high,size)
print(np.random.uniform(low=5, high=10, size=(5, 3)))

# 问题：只打印或显示numpy数组rand_arr的小数点后3位。
# 给定：
rand_arr = np.random.random((5, 3))
# 通过设置numpy的打印选项来实现
np.set_printoptions(precision=3)
print(rand_arr)

# 问题：不通过e式科学记数法来打印rand_arr（如1e10）
# 给定：
np.random.seed(100)
rand_arr = np.random.random([3, 3]) / 1e3
print(rand_arr)
# 设置suppress为False表示使用科学计数法打印
np.set_printoptions(suppress=False)
print(rand_arr)
np.set_printoptions(suppress=True, precision=6)
print(rand_arr)

# 问题：将numpy数组a中打印的项数限制为最多6个元素。
# 给定：
a = np.arange(15)
np.set_printoptions(threshold=6)
print(a)

# 问题：打印完整的numpy数组a而不截断。
np.set_printoptions(threshold=np.nan)
print(a)

# 问题：导入鸢尾属植物数据集，保持文本不变。
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
print(iris[:3])

# 问题：从前面问题中导入的一维鸢尾属植物数据集中提取文本列的物种。
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None, encoding=None)
species = np.array([row[4] for row in iris_1d])
print(species[:5])

# 问题：通过省略鸢尾属植物数据集种类的文本字段，将一维鸢尾属植物数据集转换为二维数组iris_2d。
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None, encoding=None)
# 方法1
iris_2d = np.array([row.tolist()[:4] for row in iris_1d])
print(iris_2d[:4])
# 方法2
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
print(iris_2d[:4])

# 问题：求出鸢尾属植物萼片长度的平均值、中位数和标准差(第1列)
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
mean, median, std = np.mean(sepallength), np.median(sepallength), np.std(sepallength)
print(mean, median, std)

# 问题：创建一种标准化形式的鸢尾属植物间隔长度，其值正好介于0和1之间，这样最小值为0，最大值为1。
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
# 答案：区间缩放法
Smax, Smin = np.max(sepallength), np.min(sepallength)
S = (sepallength - Smin) / (Smax - Smin)
print(S[:5])
# or numpy.ptp() 该函数返回沿轴的值的范围（最大值– 最小值）。
S = (sepallength - Smin) / sepallength.ptp()
print(S[:5])

# 问题：计算sepallength的softmax分数。
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
sepallength_exp = np.exp(sepallength)
print((sepallength_exp / sepallength_exp.sum())[:5])

# 问题：找到鸢尾属植物数据集的第5和第95百分位数
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
print(np.percentile(sepallength, q=[5, 95]))

# 问题：在iris_2d数据集中的20个随机位置插入np.nan值
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
# 方法1,i和j都是大小为600的一维数组
np.random.seed(100)
# np.random.choice中有一个replace参数来控制采样后是否有重复
# 默认replace=True，为有重复
# 先从600个位置中挑出20个
t = np.random.choice(600, 550)
iris_2d[t // 4, t % 4] = np.nan
print(np.where(np.isnan(iris_2d)))  # 发现只有354个为nan
# 将replace设置为False
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
t = np.random.choice(600, 550, replace=False)
iris_2d[t // 4, t % 4] = np.nan
print(np.where(np.isnan(iris_2d)))  # 此时有550个为nan

# 方法2，与使用choice方法时replace参数设置为True一样，该方法也无法保证随机位置的不重复性
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
np.random.seed(100)
iris_2d[np.random.randint(150, size=550), np.random.randint(4, size=550)] = np.nan
print(np.where(np.isnan(iris_2d)))  # 此时有372个为nan

# 问题：在iris_2d的sepallength中查找缺失值的数量和位置（第1列）
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
np.random.seed(100)
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
print(np.isnan(iris_2d[:, 0]).sum())
print(np.where(np.isnan(iris_2d[:, 0])))

# 问题：过滤具有petallength（第3列）> 1.5 和 sepallength（第1列）< 5.0 的iris_2d行
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
print(iris_2d[(iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)])

# 问题：选择没有任何nan值的iris_2d行。
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
# 没有直接的numpy函数能实现
# 方法1
any_nan_in_row = np.array([not np.any(np.isnan(row)) for row in iris_2d])
print(iris_2d[any_nan_in_row][:5])
# 方法2
print(iris_2d[np.sum(np.isnan(iris_2d), axis=1) == 0][:5])

# 问题：在iris_2d中找出SepalLength（第1列）和PetalLength（第3列）之间的相关性
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
# 方法1
print(np.corrcoef(iris_2d[:, 0], iris_2d[:, 2])[0, 1])
# 方法2
from scipy.stats.stats import pearsonr

corr, p_value = pearsonr(iris_2d[:, 0], iris_2d[:, 2])
print(corr)

# 问题：找出iris_2d是否有任何缺失值。
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
print(np.sum(np.isnan(iris_2d)) != 0)
print(np.isnan(iris_2d).any())

# 问题：在numpy数组中将所有出现的nan替换为0
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
print(np.where(np.isnan(iris_2d))[0].size)
iris_2d[np.isnan(iris_2d)] = 0
print(np.where(np.isnan(iris_2d))[0].size)

# 问题：找出鸢尾属植物物种中的独特值和独特值的数量
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

print(np.unique(iris[:, -1], return_counts=True))

# 问题40：将iris_2d的花瓣长度（第3列）加入以形成文本数组，这样如果花瓣长度为：
# Less than 3 --> 'small'
# 3-5 --> 'medium'
# '>=5 --> 'large'
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
# 答案
petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])
label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
petal_length_cate = [label_map[x] for x in petal_length_bin]

print(petal_length_cate[:4])

# 问题：在iris_2d中为卷创建一个新列，其中volume是（pi x petallength x sepal_length ^ 2）/ 3
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

petal_length = iris_2d[:, 2].astype('float')
sepal_length = iris_2d[:, 0].astype('float')
volume = (np.pi * petal_length * (sepal_length ** 2)) / 3
# 给volume增加一个新的维度
volume = volume[:, np.newaxis]

iris_2d = np.hstack([iris_2d, volume])
print(iris_2d[:5])

# 问题：随机抽鸢尾属植物的种类，使得刚毛的数量是云芝和维吉尼亚的两倍
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
species = iris[:, 4]
# 方法1
np.random.seed(100)
a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])
print()
# 方法2:这个采样方式高级
np.random.seed(100)
probs = np.r_[np.linspace(0, 0.500, num=50), np.linspace(0.501, 0.750, num=50), np.linspace(0.751, 1.000, num=50)]
# searchsorted为查找v在a中的位置，并返回索引值
index = np.searchsorted(probs, np.random.random(150))
species_out = species[index]
print(np.unique(species_out, return_counts=True))

# 问题：第二长的物种setosa的价值是多少
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# 先取出setosa物种的长度列
setosa_petal_length = iris[iris[:, 4] == b'Iris-setosa'][:, 2].astype('float')
print(np.unique(np.sort(setosa_petal_length))[-2])

# 问题：根据sepallength列对虹膜数据集进行排序。
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# argsort()返回的是数组值从小到大的索引值
print(iris[iris[:, 0].argsort()][:5])

# 问题：在鸢尾属植物数据集中找到最常见的花瓣长度值（第3列）。
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# 方法1
t = np.array(np.unique(iris[:, 2], return_counts=True)).T
print(t[t[:, 1].argsort()][-1, 0])
# 方法2
vals, counts = np.unique(iris[:, 2], return_counts=True)
# argmax返回沿轴axis最大值的索引
print(vals[np.argmax(counts)])

# 问题：在虹膜数据集的petalwidth第4列中查找第一次出现的值大于1.0的位置。
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
# np.argwhere返回满足条件的数组元组的索引
print(np.argwhere(iris[:, 3].astype('float') > 1.0)[0])

# 问题：从数组a中，替换所有大于30到30和小于10到10的值。
# 给定：
np.random.seed(100)
a = np.random.uniform(1, 50, 20)
# 方法1
a[a > 30] = 30
a[a < 10] = 10
print(a)
# 方法2
print(np.clip(a, a_min=10, a_max=30))
# 方法3
print(np.where(a < 10, 10, np.where(a > 30, 30, a)))

# 问题：获取给定数组a中前5个最大值的位置。
# 给定：
np.random.seed(100)
a = np.random.uniform(1, 50, 20)
# 方法1:对前n个值也进行了排序
print(a.argsort()[-10:])
# 方法2：np.argpartition，未对前n个值进行排序，效率高
print(np.argpartition(-a, 10)[:10])
# 下面几个方法可以得到前5个最大值的值
# 方法1
print(a[a.argsort()][-5:])
# 方法2
print(np.sort(a)[-5:])
# 方法3
print(np.partition(a, kth=-5)[-5:])
# 方法4
print(a[np.argpartition(-a, 5)[:5]])

# 问题：按行计算唯一值的计数。
# 给定：
np.random.seed(100)
arr = np.random.randint(1, 11, size=(6, 10))
# 先统计出每一行的唯一值以及取值数，再生成最终的结果
num_counts_list = [np.unique(row, return_counts=True) for row in arr]
result = np.array([[int(b[a == i]) if i in a else 0 for i in np.unique(arr)] for a, b in num_counts_list])
print(np.arange(1, 11))
print(result)

# 问题：将array_of_arrays转换为扁平线性1d数组。
# 给定：
arr1 = np.arange(3)
arr2 = np.arange(3, 7)
arr3 = np.arange(7, 10)

array_of_arrays = np.array([arr1, arr2, arr3])
# 方法1
arr_1d = np.array([a for arr in array_of_arrays for a in arr])
print(arr_1d)
# 方法2
arr_1d = np.concatenate(array_of_arrays)
print(arr_1d)

# 问题：计算一次性编码(数组中每个唯一值的虚拟二进制变量)
# 给定：
np.random.seed(101)
arr = np.random.randint(1, 4, size=6)
# 先得到unique
vals = np.unique(arr)
result = np.array([[1.0 if i == v else 0.0 for v in vals] for i in arr])
print(result)

# 问题：创建按分类变量分组的行号。使用以下来自鸢尾属植物物种的样本作为输入。
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
species_small = np.sort(np.random.choice(species, size=20))
print(species_small)
# 方法1
vals, counts = np.unique(species_small, return_counts=True)
result = np.concatenate([np.arange(c) for c in counts])
print(result)
# 方法2
print([i for x in np.unique(species_small, return_counts=True)[1] for i in np.arange(x)])

# 问题：根据给定的分类变量创建组ID。使用以下来自鸢尾属植物物种的样本作为输入。
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
species_small = np.sort(np.random.choice(species, size=20))
print(species_small)
print([j for i, count in enumerate(np.unique(species_small, return_counts=True)[1]) for j in np.full(count, i)])

# 问题：为给定的数字数组a创建排名。
# 给定：
np.random.seed(10)
a = np.random.randint(20, size=10)
print(a)
# 很巧妙
print(a.argsort().argsort())

# 问题：创建与给定数字数组a相同形状的排名数组。
# 给定：
np.random.seed(10)
a = np.random.randint(20, size=[2, 5])
print(a)
'''
numpy中的ravel()、flatten()、squeeze()都有将多维数组转换为一维数组的功能，区别： 
ravel()：如果没有必要，不会产生源数据的副本 
flatten()：返回源数据的副本 
squeeze()：只能对维数为1的维度降维
'''
print(a.ravel().argsort().argsort().reshape(a.shape))

# 问题：计算给定数组中每行的最大值。
# 给定：
np.random.seed(100)
a = np.random.randint(1, 10, [5, 3])
print(np.max(a, axis=1))

# 问题：为给定的二维numpy数组计算每行的最小值。
# 给定：
np.random.seed(100)
a = np.random.randint(1, 10, [5, 3])
print(np.min(a, axis=1))
print(np.apply_along_axis(np.min, axis=1, arr=a))

# 问题：在给定的numpy数组中找到重复的条目(第二次出现以后)，并将它们标记为True。第一次出现应该是False的。
# 给定：
np.random.seed(100)
a = np.random.randint(0, 5, 10)
print('Array: ', a)
# 方法1
result = np.full(a.shape, True, dtype=bool)
indexs = [int(np.argwhere(a == val)[0]) for val in np.unique(a)]
result[indexs] = False
print(result)
# 方法2：利用np.unique的return_index参数
result = np.full(a.shape, True, dtype=bool)
indexs = np.unique(a, return_index=True)[1]
result[indexs] = False
print(result)

# 问题：在二维数字数组中查找按分类列分组的数值列的平均值
# 给定：
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

group_col = iris[:, 4]
numeric_col = iris[:, 1].astype('float')
print([[val, numeric_col[group_col == val].mean()] for val in np.unique(group_col)])

# 问题：从以下URL导入图像并将其转换为numpy数组。
# 给定：
URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
# 答案
# from io import BytesIO
# from PIL import Image
# import PIL, requests
#
# # Import image from URL
# URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
# response = requests.get(URL)
#
# # Read it as Image
# I = Image.open(BytesIO(response.content))
#
# # Optionally resize
# I = I.resize([150, 150])
#
# # Convert to numpy array
# arr = np.asarray(I)
#
# # Optionaly Convert it back to an image and show
# im = PIL.Image.fromarray(np.uint8(arr))
# Image.Image.show(im)

# 问题：从一维numpy数组中删除所有NaN值
# 给定：
a = np.array([1, 2, 3, np.nan, 5, 6, 7, np.nan])
print(a[~np.isnan(a)])  # ~可以用于布尔取反

# 问题：计算两个数组a和数组b之间的欧氏距离。
# 给定：
a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7, 8])
# 方法1
print(np.sqrt(np.sum(np.square(a - b))))
# 方法2
print(np.linalg.norm(a - b))

# 问题：找到一个一维数字数组a中的所有峰值。峰顶是两边被较小数值包围的点。
# 给定：
a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
# np.diff计算相邻元素的差值
# np.sign就是大于0的返回1.0,小于0的返回-1.0,等于0的返回0.0
# np.where和np.argwhere的作用类似，返回值格式有区别
doublediff = np.diff(np.sign(np.diff(a)))
peak_locations = np.where(doublediff == -2)[0] + 1
print(peak_locations)

# 问题：从2d数组a_2d中减去一维数组b_1D，使得b_1D的每一项从a_2d的相应行中减去。
# 给定：
a_2d = np.array([[3, 3, 3], [4, 4, 4], [5, 5, 5]])
b_1d = np.array([1, 2, 3])
print(a_2d - b_1d[:, None])  # None用于改变数组的维度

# 问题：找出x中数字1的第5次重复的索引。
# 给定：
x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
print(np.where(x == 1)[0][4])

a = [1, 2, 3]
# print([i if i == 1 for i in a])  # 错误
print([i for i in a if i == 1])  # 正确

# 问题：将numpy的datetime64对象转换为datetime的datetime对象
# 给定：
dt64 = np.datetime64('2018-02-25 22:10:10')
from datetime import datetime

# 方法1:这个有点神奇
print(dt64.tolist())
# 方法2
print(dt64.astype(datetime))

# 问题：对于给定的一维数组，计算窗口大小为3的移动平均值。
# 给定：
np.random.seed(100)
Z = np.random.randint(10, size=10)
# 方法1
print([np.mean(Z[i:i + 3]) for i in range(Z.size - 2)])
# 方法2:用卷积函数666
print(np.convolve(Z, np.ones(3) / 3, mode='valid'))

# 问题：创建长度为10的numpy数组，从5开始，在连续的数字之间的步长为3。
length = 10
start = 5
step = 3
print(np.array([start + i * step for i in range(length)]))

# 问题：给定一系列不连续的日期序列。填写缺失的日期，使其成为连续的日期序列。
# 给定：
dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)
print(dates)
# 很巧妙
filled_in = np.array([np.arange(date, date + d) for date, d in zip(dates, np.diff(dates))]).reshape(-1)
print(np.hstack([filled_in, dates[-1]]))

# 问题：从给定的一维数组arr中，利用步进生成一个二维矩阵，窗口长度为4，步距为2，类似于 [[0,1,2,3], [2,3,4,5], [4,5,6,7]..]
# 给定：
arr = np.arange(15)
windows_size = 4
step = 2
# 方法1
print(np.array([arr[i:i + 4] for i in range(0, arr.size - 3, 2)]))


# 方法2
def gen_strides(a, stride_len=5, window_len=5):
    n_strides = ((a.size - window_len) // stride_len) + 1
    return np.array([a[i:i + window_len] for i in np.arange(0, n_strides * stride_len, stride_len)])


print(gen_strides(arr, stride_len=2, window_len=4))
