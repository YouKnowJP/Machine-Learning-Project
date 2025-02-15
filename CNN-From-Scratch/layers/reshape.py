import numpy as np


class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input_data):
        self.input = input_data
        return np.reshape(input_data, self.output_shape)

    def backward(self, output_gradient, learning_rate=None):
        return np.reshape(output_gradient, self.input_shape)
