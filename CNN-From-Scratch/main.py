import numpy as np
from dataset import load_data
from layers.convolutional import Convolutional
from layers.fully_connected import FullyConnected
from layers.activation import ReLU, Softmax
from loss.categorical_cross_entropy import categorical_cross_entropy, categorical_cross_entropy_prime

# Load dataset
X_train, X_test, y_train, y_test = load_data()

# Initialize layers
conv1 = Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=8)
relu1 = ReLU()
fc1 = FullyConnected(input_size=8 * 26 * 26, output_size=128)
relu2 = ReLU()
fc2 = FullyConnected(input_size=128, output_size=10)
softmax = Softmax()

# Training parameters
epochs = 5
learning_rate = 0.01
batch_size = 32

# Training loop
for epoch in range(epochs):
    loss = 0
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        batch_loss = 0
        grad = np.zeros_like(y_batch)

        for j in range(len(X_batch)):
            x = X_batch[j]
            y = y_batch[j]

            # Forward pass
            conv_out = conv1.forward(x)
            relu_out = relu1.function(conv_out)
            fc1_out = fc1.forward(relu_out.flatten())
            relu2_out = relu2.function(fc1_out)
            fc2_out = fc2.forward(relu2_out)
            y_pred = softmax.forward(fc2_out)

            # Compute loss
            batch_loss += categorical_cross_entropy(y, y_pred)
            grad[j] = categorical_cross_entropy_prime(y, y_pred)

            # Backward pass
            grad_fc2 = fc2.backward(grad[j], learning_rate)
            grad_relu2 = relu2.derivative(relu2_out) * grad_fc2
            grad_fc1 = fc1.backward(grad_relu2, learning_rate)
            grad_conv = relu1.derivative(relu_out) * grad_fc1.reshape(conv1.output_shape)
            conv1.backward(grad_conv, learning_rate)

        loss += batch_loss

    print(f"Epoch {epoch + 1}, Loss: {loss / len(X_train)}")

print("Training Complete!")
