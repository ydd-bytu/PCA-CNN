import numpy as np


class LagrangeNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 随机初始化权重
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

        # 初始化 Lagrange 乘子
        self.lagrange_multiplier = np.zeros((hidden_size, output_size))

    def forward(self, inputs):
        # 前向传播
        hidden_inputs = np.dot(inputs, self.weights_input_hidden)
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = self.sigmoid(final_inputs)

        return final_outputs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backward(self, inputs, targets, learning_rate):
        # 反向传播
        hidden_inputs = np.dot(inputs, self.weights_input_hidden)
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = self.sigmoid(final_inputs)

        output_errors = targets - final_outputs
        output_delta = output_errors * self.sigmoid_derivative(final_outputs)

        hidden_errors = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_errors * self.sigmoid_derivative(hidden_outputs)

        # 更新权重
        self.weights_hidden_output += hidden_outputs.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate

    def lagrange_backward(self, inputs, targets, constraint, learning_rate):
        # Lagrange 反向传播
        hidden_inputs = np.dot(inputs, self.weights_input_hidden)
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = self.sigmoid(final_inputs)

        output_errors = targets - final_outputs
        output_delta = output_errors * self.sigmoid_derivative(final_outputs)

        hidden_errors = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_errors * self.sigmoid_derivative(hidden_outputs)

        # 更新权重
        self.weights_hidden_output += hidden_outputs.T.dot(output_delta) * learning_rate

        # 更新 Lagrange 乘子
        self.lagrange_multiplier += constraint * hidden_outputs * learning_rate

        # 更新输入层到隐藏层的权重
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate \
                                     + self.lagrange_multiplier.dot(self.weights_hidden_output.T) * learning_rate


# 示例用法
input_size = 2
hidden_size = 3
output_size = 1

# 初始化网络
network = LagrangeNeuralNetwork(input_size, hidden_size, output_size)

# 输入数据
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# 定义约束条件
constraint = np.array([[1], [1], [1]])

# 训练网络
for _ in range(10000):
    network.lagrange_backward(inputs, targets, constraint, learning_rate=0.1)

# 进行预测
predictions = network.forward(inputs)
print("预测结果：")
print(predictions)