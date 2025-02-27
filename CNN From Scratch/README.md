# **CNN From Scratch - Digit Recognizer**  

This project implements a **Convolutional Neural Network (CNN) from scratch** to classify handwritten digits from the [Kaggle Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer). It does not use deep learning frameworks like TensorFlow or PyTorch but instead builds all layers manually, including convolutional, activation, and fully connected layers.

## **Project Structure**  

```
CNN-From-Scratch/
│── dataset.py               # Loads and preprocesses the dataset
│── main.py                  # Main script to train and evaluate the CNN
│── layers/
│   ├── convolutional.py      # Convolutional layer implementation
│   ├── fully_connected.py    # Fully connected (dense) layer
│   ├── activation.py         # Activation functions (ReLU, Softmax)
│   ├── reshape.py            # Reshape layer for transitioning between layers
│── loss/
│   ├── categorical_cross_entropy.py  # Loss function implementation
│── digit-recognizer/
│   ├── train.csv             # Training dataset from Kaggle
│   ├── test.csv              # Test dataset from Kaggle
│── README.md                 # Project documentation
```

## **Dataset**  
The dataset can be downloaded from **[Kaggle's Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer)**. It consists of:  
- **train.csv**: Contains 42,000 grayscale images (28x28 pixels) with labels (digits 0-9).  
- **test.csv**: Contains 28,000 grayscale images without labels (used for submission).  

## **Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/CNN-From-Scratch.git
cd CNN-From-Scratch
```

### **2. Install Dependencies**  
Ensure you have Python 3.8+ and install the required packages:  
```bash
pip install -r requirements.txt
```

### **3. Download the Dataset**  
Place the `train.csv` and `test.csv` files into the `digit-recognizer/` directory.  

### **4. Run the Model**  
Train and evaluate the CNN using:  
```bash
python main.py
```

## **Model Architecture**  
This CNN is built manually and consists of:  
1. **Convolutional Layer (3x3 kernel, 8 filters)**  
2. **ReLU Activation**  
3. **Reshape Layer (Flattening for Fully Connected Layer)**  
4. **Fully Connected Layer (128 neurons, ReLU Activation)**  
5. **Fully Connected Output Layer (10 neurons, Softmax Activation)**  

## **Results**  
The model achieves **~98% accuracy on the validation set** when trained for five epochs.

## **License**  
This project is open-source under the **MIT License**.
