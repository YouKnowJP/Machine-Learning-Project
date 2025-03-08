# **Linear Regression from Scratch**  

This project implements **Linear Regression** using **Gradient Descent** from scratch without using libraries like `scikit-learn`. We will derive the regression line equation and iteratively optimize it using gradient descent.

---

## **Overview**  
Linear Regression is a supervised learning algorithm used for predicting a continuous target variable (`Y`) based on an independent variable (`X`). The relationship between `X` and `Y` is modeled as:

$[
Y = mX + b
]$

where:  
- $( Y )$ = Target (dependent variable)  
- $( X )$ = Feature (independent variable)  
- $( m )$ = Slope of the line  
- $( b )$ = Intercept (where the line crosses the Y-axis)

The goal is to find the best values for $( m $) and $( b $) that minimize the error between the predicted and actual values.

---

## **Dataset**  
We use a dataset (`Kaggle/score_updated.csv`) that contains:  
- `Hours`: Number of hours studied  
- `Scores`: Exam score obtained  

---

## **Loss Function**  
To measure how well the regression line fits the data, we use the **Mean Squared Error (MSE)**:

$[
MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y_i})^2
]$

where:  
- $( Y_i )$ = Actual value  
- $( \hat{Y_i} )$ = Predicted value using $( mX + b )$  
- $( n )$ = Number of data points  

The objective is to minimize this error.

---

## **Gradient Descent Algorithm**  
Gradient Descent is an optimization algorithm used to minimize the loss function by iteratively updating the parameters (`m` and `b`) using the gradient (derivative) of the loss function.

### **Gradient Formulas**
The gradient (partial derivatives) for the slope \( m \) and intercept \( b \) are:

$[
\frac{\partial J}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} X_i (Y_i - \hat{Y_i})
]$

$[
\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (Y_i - \hat{Y_i})
]$

We update the parameters using the **learning rate (α)**:

$[
m = m - \alpha \times \frac{\partial J}{\partial m}
]$

$[
b = b - \alpha \times \frac{\partial J}{\partial b}
]$

where **α** (learning rate) controls the step size.

---

## **Implementation Steps**
1. **Load Dataset**  
   - Read `score_updated.csv` using `pandas`.

2. **Define Loss Function**  
   - Compute Mean Squared Error (MSE).

3. **Implement Gradient Descent**  
   - Compute gradients for `m` and `b`.
   - Update `m` and `b` iteratively.

4. **Train the Model**  
   - Run **gradient descent** for a fixed number of epochs.

5. **Visualize the Results**  
   - Plot the original data points and the learned regression line.

---

## **Expected Output**
- The model will print `m` (slope) and `b` (intercept) updates every 50 epochs.
- The final output will be a **scatter plot** of actual data and a **best-fit regression line**.

---

## **Hyperparameters**
| Parameter        | Description                   | Default Value |
|-----------------|------------------------------|--------------|
| `learning_rate` | Step size for gradient updates | 0.01         |
| `epochs`        | Number of iterations          | 1000         |

---

## **Conclusion**
This implementation demonstrates how **Linear Regression** can be trained using **Gradient Descent** from scratch. We:
- Defined the **Loss Function (MSE)**.
- Derived **Gradients** and updated **Parameters**.
- Trained the model iteratively.
- Visualized the **Regression Line**.
