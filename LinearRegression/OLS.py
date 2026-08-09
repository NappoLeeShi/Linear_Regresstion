import numpy as np

class OLS:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.r2score = None
        self.y_pred = None

    def fit(self, X, y):
        n = X.shape[0]
        X_b = np.c_[np.ones((n, 1)), X]

        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = self.coefficients[0]

        self.y_pred = np.dot(X_b, self.coefficients)
        y_mean = np.mean(y)

        self.r2score = 1 - ( np.sum((y-self.y_pred)**2) / np.sum(y-y_mean)**2)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coefficients)


        

