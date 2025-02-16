import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import load_data
from layers.convolutional import Convolutional
from layers.fully_connected import FullyConnected
from layers.activation import ReLU, Softmax
from loss.categorical_cross_entropy import categorical_cross_entropy, categorical_cross_entropy_prime

# Configuration settings
CONFIG = {
    "epochs": 1,
    "learning_rate": 0.01,
    "batch_size": 32,
    "conv_input_shape": (1, 28, 28),
    "conv_kernel_size": 3,
    "conv_depth": 8,
    "fc1_input_size": 8 * 26 * 26,
    "fc1_output_size": 128,
    "fc2_input_size": 128,
    "fc2_output_size": 10,
    "submission_file": "submission.csv File Path"
}

# Define train model
def train_model(X_train, y_train, conv1, relu1, fc1, relu2, fc2, softmax):
    for epoch in range(CONFIG["epochs"]):
        loss = 0
        progress_bar = tqdm(range(0, len(X_train), CONFIG["batch_size"]), desc=f"Epoch {epoch + 1}")

        for i in progress_bar:
            X_batch = X_train[i:i + CONFIG["batch_size"]]
            y_batch = y_train[i:i + CONFIG["batch_size"]]

            batch_loss = 0
            grad = np.zeros_like(y_batch)

            for j in range(len(X_batch)):
                x = X_batch[j]
                y = y_batch[j]

                conv_out = conv1.forward(x)
                relu_out = relu1.function(conv_out)
                fc1_out = fc1.forward(relu_out.flatten())
                relu2_out = relu2.function(fc1_out)
                fc2_out = fc2.forward(relu2_out)
                y_pred = softmax.forward(fc2_out)

                batch_loss += categorical_cross_entropy(y, y_pred)
                grad[j] = categorical_cross_entropy_prime(y, y_pred)

                grad_fc2 = fc2.backward(grad[j], CONFIG["learning_rate"])
                grad_relu2 = relu2.derivative(relu2_out) * grad_fc2
                grad_fc1 = fc1.backward(grad_relu2, CONFIG["learning_rate"])
                grad_conv = relu1.derivative(relu_out) * grad_fc1.reshape(conv1.output_shape)
                conv1.backward(grad_conv, CONFIG["learning_rate"])

            loss += batch_loss
            progress_bar.set_postfix(loss=loss / (i + CONFIG["batch_size"]))

        print(f"Epoch {epoch + 1}, Loss: {loss / len(X_train)}")
    print("Training Complete!")

# Define predict model
def predict(X_test, conv1, relu1, fc1, relu2, fc2, softmax):
    predictions = []
    for x in tqdm(X_test, desc="Predicting"):
        conv_out = conv1.forward(x)
        relu_out = relu1.function(conv_out)
        fc1_out = fc1.forward(relu_out.flatten())
        relu2_out = relu2.function(fc1_out)
        fc2_out = fc2.forward(relu2_out)
        y_pred = softmax.forward(fc2_out)
        predictions.append(np.argmax(y_pred))
    return predictions

# Run main function
def main():
    # Load dataset
    X_train, y_train, X_test = load_data()

    # Initialize layers
    conv1 = Convolutional(input_shape=CONFIG["conv_input_shape"], kernel_size=CONFIG["conv_kernel_size"], depth=CONFIG["conv_depth"])
    relu1 = ReLU()
    fc1 = FullyConnected(input_size=CONFIG["fc1_input_size"], output_size=CONFIG["fc1_output_size"])
    relu2 = ReLU()
    fc2 = FullyConnected(input_size=CONFIG["fc2_input_size"], output_size=CONFIG["fc2_output_size"])
    softmax = Softmax()

    # Train model
    train_model(X_train, y_train, conv1, relu1, fc1, relu2, fc2, softmax)

    # Make predictions
    predictions = predict(X_test, conv1, relu1, fc1, relu2, fc2, softmax)

    # Save predictions
    submission = pd.DataFrame({"ImageId": np.arange(1, len(predictions) + 1), "Label": predictions})
    submission.to_csv(CONFIG["submission_file"], index=False)
    print(f"Predictions saved to {CONFIG['submission_file']}")

if __name__ == "__main__":
    main()
