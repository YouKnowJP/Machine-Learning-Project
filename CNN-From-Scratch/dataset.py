import numpy as np
import pandas as pd

# Loading dataset
def load_data():
    train_df = pd.read_csv(
        "Train Data.csv File Path")
    X_train = train_df.drop(columns=["label"]).values
    y_train = train_df["label"].values

    test_df = pd.read_csv(
        "Test Data.csv File Path")
    X_test = test_df.values

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    num_classes = 10
    y_train_one_hot = np.zeros((y_train.size, num_classes))
    y_train_one_hot[np.arange(y_train.size), y_train] = 1

    return X_train, y_train_one_hot, X_test
