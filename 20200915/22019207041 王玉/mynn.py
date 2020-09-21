#导入numpy包
import numpy as np
    # 定义激活函数 f(x) = 1 / (1 * e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
#定义神经网络表达式
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

#设定输入节点x(n=4)个,输出节点y(m=5)个
weights = np.array([0.2,0.5,0.7,0.8]) # w1 =0.2,w2=0.5,w3=0.7,w4=0.8
bias= np.array([4,5,6,7,8])
n = Neuron(weights, bias)

x = np.array([1,2,3,4])   # x1 = 1, x2 = 2, x3 = 3, x4 = 4
#输出y
y=print(n.feedforward(x)) #y = [0.99997246 0.99998987 0.99999627 0.99999863 0.9999995 ]
y
#print(numpy.around(n.feedforward(x), decimals=16)）