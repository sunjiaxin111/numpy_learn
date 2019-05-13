import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


def load_data():
    data = np.loadtxt('Data/sklearn_digits.csv', delimiter=',')

    # first ten values are the one hot encoded y (target) values
    y = data[:, 0:10]

    data = data[:, 10:]  # x data
    # data = data - data.mean(axis = 1)
    data -= data.min()  # scale the data so values are between 0 and 1
    data /= data.max()  # scale

    out = []

    # populate the tuple list with the data
    for i in range(data.shape[0]):
        fart = list((data[i, :].tolist(), y[i].tolist()))  # don't mind this variable name
        out.append(fart)

    return out


class MLP_NeuralNetwork(object):
    def __init__(self, input, hidden, output):
        '''
        :param input: 输入层神经元的个数
        :param hidden: 隐藏层神经元的个数
        :param output: 输出层神经元的个数
        '''
        self.input = input + 1  # 加1作为偏置节点
        self.hidden = hidden
        self.output = output
        # 设置激活占位符
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output
        # 创建随机权重,randn函数返回一组具有标准正态分布的数据
        self.wi = np.random.randn(self.input, self.hidden)
        self.wo = np.random.randn(self.hidden, self.output)

    def feedForward(self, inputs):
        if len(inputs) != self.input - 1:
            raise ValueError('输入有误！')
        # 输入层激活
        for i in range(self.input - 1):
            self.ai[i] = inputs[i]
        # 隐藏层激活
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i, j]
            self.ah[j] = sigmoid(sum)
        # 输出层激活
        for k in range(self.output):
            sum = 0.0
            for i in range(self.hidden):
                sum += self.ah[i] * self.wo[i, k]
            self.ao[k] = sigmoid(sum)

        return self.ao

    def backPropagate(self, targets, N):
        '''
        :param targets: 标签值
        :param N: 学习速率
        :return: 更新的权重值，当前误差
        '''
        if len(targets) != self.output:
            raise ValueError('输入错误！')
        # 计算output_deltas
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = self.ao[k] - targets[k]  # 这个是损失函数对预测值的导数
            output_deltas[k] = dsigmoid(self.ao[k]) * error
        # 计算hidden_deltas
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j, k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
        # 更新隐藏层到输出层的权重
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j, k] -= N * change
        # 更新输入层到隐藏层的权重
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i, j] -= N * change
        # 计算误差
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (self.ao[k] - targets[k]) ** 2

        return error

    def train(self, patterns, iterations=3000, N=0.0002):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error = self.backPropagate(targets, N)
            if i % 500 == 0:
                print('error %-.5f' % error)

    def predict(self, X):
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))

        return predictions


if __name__ == '__main__':
    X = load_data()
    NN = MLP_NeuralNetwork(64, 100, 10)
    NN.train(X)
