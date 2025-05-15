import numpy as np
from preprocessing import preprocess_data

# Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# KNN implementation
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, x_test):
        distances = []
        for i in range(len(self.X_train)):
            dist = euclidean_distance(self.X_train[i], x_test)
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest = [label for _, label in distances[:self.k]]
        prediction = max(set(k_nearest), key=k_nearest.count)
        return prediction

# Train KNN and return the model
def train_and_return_knn():
    X_train, X_test, y_train, y_test = preprocess_data("../data/ai_art", "../data/human_art")
    model = KNN(k=3)
    model.fit(X_train, y_train)
    return model

# Run directly
if __name__ == "__main__":
    model = train_and_return_knn()
    print("KNN model trained and ready.")
