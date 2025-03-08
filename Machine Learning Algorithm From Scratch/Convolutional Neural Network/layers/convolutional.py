import numpy as np
from scipy import signal

# Define convolutional layer
class Convolutional:
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.kernel_size = kernel_size

        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        self.kernels = np.random.randn(*self.kernels_shape) * 0.1
        self.biases = np.random.randn(*self.output_shape) * 0.1

    def forward(self, input_data):
        self.input = input_data
        self.output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], mode="valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], mode="valid")
                input_gradient[j] += signal.correlate2d(output_gradient[i], self.kernels[i, j], mode="full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
