import numpy as np

arr = np.arange(36).reshape(3, 4, 3)
print(arr)
print(arr.shape)

'''
这种用数组表达式替换显式循环的做法通常称为向量化。
在Python中循环数组或任何数据结构时，会涉及很多开销。 
NumPy中的向量化操作将内部循环委托给高度优化的C和Fortran函数，从而实现更清晰，更快速的Python代码。
'''

# 考虑一个True和False的一维向量，你要为其计算序列中“False to True”转换的数量：
# 比较使用for循环和矢量化的性能差异
np.random.seed(444)
x = np.random.choice([False, True], size=100000)
print(x)


# 使用for循环
def count_transitions(x):
    count = 0
    for i, j in zip(x[:-1], x[1:]):
        if not i and j:
            count += 1

    return count


print(count_transitions(x))

# 矢量化,count_nonzero返回数组的非零个数
print(np.count_nonzero(x[:-1] < x[1:]))

# 比较两者的性能
# from timeit import timeit

setup = 'from __main__ import count_transitions, x; import numpy as np'
num = 1000
# t1 = timeit('count_transitions(x)', setup=setup, number=num)
# t2 = timeit('np.count_nonzero(x[:-1] < x[1:])', setup=setup, number=num)
# print('执行速度差异:%.1fx' % (t1 / t2))

'''
假定一只股票的历史价格是一个序列，假设你只允许进行一次购买和一次出售，
那么可以获得的最大利润是多少？例如，假设价格=(20，18，14，17，20，21，15)，
最大利润将是7，从14买到21卖。
'''


# 一个O(n)复杂度的解决方案
def profit(prices):
    max_px = 0
    min_px = prices[0]
    for px in prices[1:]:
        min_px = min(px, min_px)
        max_px = max(max_px, px - min_px)

    return max_px


prices = [20, 18, 14, 17, 20, 21, 15]
print(profit(prices))

# 这可以用NumPy实现吗？行!没问题。但首先，让我们构建一个准现实的例子：
prices = np.full(100, fill_value=np.nan)
prices[[0, 25, 60, -1]] = [80., 30., 75., 50.]
x = np.arange(len(prices))
is_valid = ~np.isnan(prices)
# 对缺失值进行线性插值
prices = np.interp(x=x, xp=x[is_valid], fp=prices[is_valid])
# 加噪声
prices += np.random.randn(len(prices)) * 2

# 下面是matplotlib的示例。俗话说：买低(绿)，卖高(红)：
import matplotlib.pyplot as plt

# 这不是一个完全正确的解法，当最小值点在最大值点之后就会出问题
# argmin取最小值的下标
mn = np.argmin(prices)
mx = mn + np.argmax(prices[mn:])
kwargs = {'markersize': 12, 'linestyle': '', 'marker': 'o'}

fig, ax = plt.subplots()
ax.plot(prices)
ax.set_title('Price History')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.plot(mn, prices[mn], color='green', **kwargs)
ax.plot(mx, prices[mx], color='red', **kwargs)
# plt.show()

# NumPy实现是什么样的？ 虽然没有np.cummin() “直接”，
# 但NumPy的通用函数（ufuncs）都有一个accumulate()方法，它的名字暗示了：
cummin = np.minimum.accumulate


# 从纯Python示例扩展逻辑，你可以找到每个价格和
# 运行最小值（元素方面）之间的差异，然后获取此序列的最大值

def profit_with_numpy(prices):
    # price减去累积的最小价格
    return np.max(prices - cummin(prices))


print(profit_with_numpy(prices))
print(profit(prices))
# allclose用于检验两者是否相同
print(np.allclose(profit_with_numpy(prices), profit(prices)))

# 比较性能，采取更长的序列
# from timeit import timeit

seq = np.random.randint(0, 100, size=100000)
print(seq)
setup = 'from __main__ import profit_with_numpy, profit, seq; import numpy as np'
num = 250
# pytime = timeit('profit(seq)', setup=setup, number=num)
# nptime = timeit('profit_with_numpy(seq)', setup=setup, number=num)
# print('执行速度差异:%.1fx' % (pytime / nptime))

# Intermezzo：理解轴符号
arr = np.array([[1, 2, 3],
                [10, 20, 30]])
print(arr.sum(axis=0))
print(arr.sum(axis=1))

'''
AXIS关键字指定将折叠的数组的维度，而不是将要返回的维度。
因此，指定Axis=0意味着第一个轴将折叠：
对于二维数组，这意味着每列中的值将被聚合。
'''
# 广播
# 两个NumPy数组(大小相等)之间的操作是按元素操作的
a = np.array([1.5, 2.5, 3.5])
b = np.array([10., 5., 1.])
print(a / b)

# 让我们以一个例子为例，我们想要减去数组的每个列的平均值，元素的平均值
# np.random.normal(loc=0, scale=1, size)对应标准正太分布
sample = np.random.normal(loc=[2., 20.], scale=[1., 3.5], size=(3, 2))
print(sample)
mu = sample.mean(axis=0)
print(mu)
print('sample:', sample.shape, '|means:', mu.shape)
print(sample - mu)

# 技术细节：较小的数组或标量不是按字面意义上在内存中展开的：重复的是计算本身。
# 扩展到标准化每列
print((sample - sample.mean(axis=0)) / sample.std(axis=0))
# 当形状较小的数组在目前的形状下无法伸展时，需要扩展其维度
# 注意: [:, None]是一种扩展数组维度的方法，用于创建长度为1的轴。np.newaxis是None的别名。
print(sample.min(axis=1)[:, None])
print(sample - sample.min(axis=1)[:, None])

'''
如果以下规则产生有效结果，则一组数组被称为“可广播”到相同的形状，这意味着 以下之一为真 时：
1、矩阵都具有完全相同的形状。
2、矩阵都具有相同数量的维度，每个维度的长度是公共长度或1。
3、具有太少尺寸的矩列可以使其形状前面具有长度为1的尺寸以满足属性＃2。
'''

# 假设你有以下四个数组：
a = np.sin(np.arange(10)[:, None])
b = np.random.randn(1, 10)
# 返回与a相同类型和形态，并且用fill_value的值填充的数组
c = np.full_like(a, 10)
d = 8
# 在检查形状之前，NumPy首先将标量转换为具有一个元素的数组：
arrays = [np.atleast_1d(arr) for arr in (a, b, c, d)]
for arr in arrays:
    print(arr.shape)

# 现在我们可以检查标准＃1。
# 如果所有数组具有相同的形状，则它们的一组形状将缩减为一个元素，
# 因为set() 构造函数有效地从其输入中删除重复项。这里显然没有达到这个标准：
print(len(set(arr.shape for arr in arrays)) == 1)
# 标准＃2的第一部分也失败了，这意味着整个标准失败：
print(len(set(arr.ndim for arr in arrays)) == 1)
# 标准＃3：具有太少尺寸的矩列可以使其形状前面具有长度为1的尺寸以满足属性＃2。
# 为了对此进行编码，你可以首先确定最高维数组的维度，然后将其添加到每个形状元组，直到所有数组具有相同的维度
maxdim = max(arr.ndim for arr in arrays)
shapes = np.array([(1,) * (maxdim - arr.ndim) + arr.shape for arr in arrays])
print(shapes)
# 最后，你需要测试每个维度的长度是否是公共长度，或是1。
# 这样做的一个技巧是首先在“等于”的位置屏蔽“shape-tuples”数组。
# 然后，你可以检查 peak-to-peak（np.ptp()）列方差是否都为零
masked = np.ma.masked_where(shapes == 1, shapes)
print(np.all(masked.ptp(axis=0) == 0))  # ptp: max - min
# 尝试把a和b相加
print(a.shape)
print(b.shape)
print((a + b).shape)


# 把逻辑封装在单个函数中
def can_broadcast(*arrays):
    arrays = [np.atleast_1d(arr) for arr in arrays]
    if len(set(arr.shape for arr in arrays)) == 1:
        return True
    if len(set((arr.ndim) for arr in arrays)) == 1:
        return True
    maxdim = max(arr.ndim for arr in arrays)
    shapes = np.array([(1,) * (maxdim - arr.ndim) + arr.shape for arr in arrays])
    masked = np.ma.masked_where(shapes == 1, shapes)
    return np.all(masked.ptp(axis=0) == 0)


print(can_broadcast(a, b, c, d))


# 幸运的是，你可以选择一个快捷方式并使用np.cast()来进行这种健全性检查，尽管它并不是为此目的而显式设计的：
def can_broadcast(*arrays):
    try:
        np.broadcast(*arrays)
        return True
    except ValueError:
        return False


print(can_broadcast(a, b, c, d))

# 矩阵编程实际应用：示例
# 聚类算法
# 机器学习是一个可以经常利用矢量化和广播的领域。 假设你有三角形的顶点（每行是x，y坐标）：
tri = np.array([[1, 1],
                [3, 1],
                [2, 3]])

# 这个“簇”的质心是(x, y)坐标，它是每列的算术平均值：
centroid = tri.mean(axis=0)
print(centroid)

# 可视化
trishape = plt.Polygon(tri, edgecolor='r', alpha=0.2, lw=5)
_, ax = plt.subplots(figsize=(4, 4))
ax.add_patch(trishape)
ax.set_ylim([.5, 3.5])
ax.set_xlim([.5, 3.5])
ax.scatter(*centroid, color='g', marker='D', s=70)
ax.scatter(*tri.T, color='b', s=70)
# plt.show()

# 对于上面的三坐标集，每个点到原点(0, 0) 的欧几里德距离是：
print(np.sqrt(np.sum(tri ** 2, axis=1)))
# 也可以直接调用现成的函数
print(np.linalg.norm(tri, axis=1))
# 你也可以找到相对于三角形质心的每个点的范数，而不是参考原点：
print(np.linalg.norm(tri - centroid, axis=1))

# K-Means聚类的算法
# 下面的这个repeat生成一个10*2的数组，分别为5个[5, 5]和5个[10, 10]
X = np.repeat([[5, 5], [10, 10]], [5, 5], axis=0)
X = X + np.random.randn(*X.shape)
centroids = np.array([[5, 5], [10, 10]])
print(X)
print(centroids)

# 换句话说，我们想回答这个问题，X中的每个点所属的质心是什么？
# 为了计算X中每个点与质心中每个点之间的欧几里德距离，我们需要进行一些重构以在此处启用广播
print(centroids[:, None])
print(centroids[:, None].shape)  # 1个:代表一维，这行代码的意思是在第一维后加一个新的维度

# 这使我们能够使用一个数组行的组合乘积，从另一个数组中清清楚楚地减掉这些数组：
print(np.linalg.norm(X - centroids[:, None], axis=2).round(2))
# 找出每个点离的最近的质心的下标
labels = np.argmin(np.linalg.norm(X - centroids[:, None], axis=2), axis=0)
print(labels)

# 可视化
c1, c2 = ['#bc13fe', '#be0119']  # https://xkcd.com/color/rgb/
llim, ulim = np.trunc([X.min() * 0.9, X.max() * 1.1])
_, ax = plt.subplots(figsize=(5, 5))
ax.scatter(*X.T, c=np.where(labels, c2, c1), alpha=0.4, s=80)
ax.scatter(*centroids.T, c=[c1, c2], marker='s', s=95, edgecolor='yellow')
ax.set_ylim([llim, ulim])
ax.set_xlim([llim, ulim])
ax.set_title('One K-Means Iteration: Predicted Classes')
# plt.show()

# 摊还（分期）表
# 矢量化也适用于金融领域。
# 给定年利率，支付频率（每年的次数），初始贷款余额和贷款期限，你可以以矢量化方式创建包含月贷款余额和付款的摊还表。让我们先设置一些标量常量：
freq = 12  # 每年12个月
rate = .0675  # 年利率6.75%
nper = 30  # 30年
pv = 200000  # 贷款面值
rate /= freq  # 按月
nper *= freq  # 360个月
# NumPy预装了一些财务函数，与Excel表兄弟不同，它们能够以矢量的形式输出。
# 债务人（或承租人）每月支付一笔由本金和利息部分组成的固定金额。由于未偿还的贷款余额下降，总付款的利息部分随之下降。
periods = np.arange(1, nper + 1, dtype=int)
principal = np.ppmt(rate, periods, nper, pv)
interest = np.ipmt(rate, periods, nper, pv)
pmt = principal + interest


# 接下来，你需要计算每月的余额，包括支付前和付款后的余额，可以定义为原始余额的未来价值减去年金(支付流)的未来价值，使用折扣因子d：
# 从功能上看，如下所示：
def balance(pv, rate, nper, pmt):
    d = (1 + rate) ** nper
    return pv * d - pmt * (d - 1) / rate


# 最后，你可以使用Pandas 的 DataFrame 将其放到表格格式中。小心这里的标志。从债务人的角度看，PMT是一种流出。
import pandas as pd

cols = ['beg_bal', 'prin', 'interest', 'end_bal']
data = [balance(pv, rate, periods - 1, -pmt),
        principal,
        interest,
        balance(pv, rate, periods, -pmt)]

table = pd.DataFrame(data, columns=periods, index=cols)
table.index.name = 'month'

with pd.option_context('display.max_rows', 6):
    print(table.round(2))

# 判断30年后是否还清
final_month = periods[-1]
print(np.allclose(table.loc['end_bal', final_month], 0))
