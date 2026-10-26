# --- cell 1 ---
import numpy as np
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    m,n = X_train.shape
    w = np.zeros(n)
    b = 0.0
    learning_rate = 0.05
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    for i in range(1000):
        preds = X_train@w + b
        probs = sigmoid(preds)
        error = probs - y_train
        dw = (X_train.T @ error) / m 
        db = np.mean(error)
        w -= learning_rate * dw
        b -= learning_rate * db
    
    def predict(X):
        return (sigmoid(X @ w + b) >= 0.5).astype(int)
    
    return predict