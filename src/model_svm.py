import numpy as np
from preprocessing import preprocess_data

# SVM implementation (binary linear)
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y == 0, -1, 1)  # Convert 0 to -1 for SVM margin logic

        self.w = np.zeros(n_features)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.where(approx >= 0, 1, 0)  # Map decision boundary to original labels

# Train and return SVM model
def train_and_return_svm():
    X_train, X_test, y_train, y_test = preprocess_data("../data/ai_art", "../data/human_art")
    model = SVM(learning_rate=0.0001, lambda_param=0.01, n_iters=500)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    model = train_and_return_svm()
    print("SVM model trained and ready.")
