'''
k-means算法的基础是最小误差平方和准则。
直观的来说，各类内的样本越相似，其与该类均值间的误差平方越小
'''
import matplotlib.pyplot as plt
from numpy import *


# 计算欧式距离
def euclDistance(vector1, vector2):
    # np.sqrt(x) ： 计算数组各元素的平方根
    # np.sum(x) ： 计算数组元素和
    # np.power(x1, x2) ： 数组元素求n次方
    return sqrt(sum(power(vector2 - vector1, 2)))


# 初始化质心
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means聚类
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # 第一列储存样本属于哪个聚类中心
    # 第二列储存这个样本和所属聚类中心的误差
    # np.mat(x) ： 转化为矩阵
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    # 初始化聚类中心
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        # 遍历每一个样本
        for i in range(numSamples):
            minDist = inf
            minIndex = 0
            # 遍历每个聚类中心，以找到样本离的最近的那个聚类中心
            for j in range(k):
                distance = euclDistance(dataSet[i, :], centroids[j, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # 更新聚类信息
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        # 更新聚类中心
        for j in range(k):
            # .A指的是转化成ndarray对象
            # nonzero返回非零元素的索引
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)

    print('聚类完成！')
    return centroids, clusterAssment


# 在2维数组中可视化
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print('只能可视化二维数组！')
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print('聚类中心个数太多了！')
        return 1

    # 画每个样本
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画聚类中心点
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


if __name__ == '__main__':
    print('第一步：读取数据')
    dataSet = []
    fileIn = open('testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])

    print('第二步：聚类')
    dataSet = mat(dataSet)
    k = 4
    centroids, clusterAssment = kmeans(dataSet, k)

    print('第三步：展示结果')
    showCluster(dataSet, k, centroids, clusterAssment)
