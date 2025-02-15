import numpy as np


class Activation:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return (x > 0).astype(float)

        super().__init__(relu, relu_prime)


class Softmax:
    def forward(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def backward(self, output_gradient):
        return output_gradient
