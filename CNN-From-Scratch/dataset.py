import numpy as np
import pandas as pd


def load_data():
    """
    Load the Digit Recognizer dataset from Kaggle.

    Returns:
        X_train: Training images (num_samples, 1, 28, 28)
        y_train: One-hot encoded training labels (num_samples, 10)
        X_test: Testing images (num_samples, 1, 28, 28)
        y_test: One-hot encoded testing labels (num_samples, 10)
    """
    # Load training set
    train_df = pd.read_csv("CNN-From-Scratch/digit-recognizer/train.csv")
    X_train = train_df.drop(columns=["label"]).values
    y_train = train_df["label"].values

    # Load testing set
    test_df = pd.read_csv("CNN-From-Scratch/digit-recognizer/test.csv")
    X_test = test_df.drop(columns=["label"]).values
    y_test = test_df["label"].values

    # Normalize images to range [0,1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshape images to (num_samples, 1, 28, 28) for CNN
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    # One-hot encode labels
    num_classes = 10
    y_train_one_hot = np.zeros((y_train.size, num_classes))
    y_train_one_hot[np.arange(y_train.size), y_train] = 1

    y_test_one_hot = np.zeros((y_test.size, num_classes))
    y_test_one_hot[np.arange(y_test.size), y_test] = 1

    return X_train, y_train_one_hot, X_test, y_test_one_hot
