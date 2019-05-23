import numpy as np

if __name__ == '__main__':
    a = np.arange(12).reshape(3, 4)
    # ravel、reshape、T这三者产生的变量都指向原始数组
    b = a.ravel()
    c = a.reshape(6, 2)
    d = a.T
    b[0] = 99
    print(a)
    print(b)
    print(c)
    print(d)

    # reshape 函数返回具有修改形状的参数，而 ndarray.resize 方法修改数组本身
    a.resize((2, 6))
    print(a)

    # 将一个数组分成几个较小的数组
    a = np.floor(10 * np.random.random((2, 12)))
    print(a)
    print(np.hsplit(a, 3))
    print(np.hsplit(a, (3, 4)))  # Split a after the third and the fourth column
