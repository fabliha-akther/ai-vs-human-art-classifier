import numpy as np
from sklearn.tree import DecisionTreeClassifier
from preprocessing import preprocess_data

# Random Forest implementation
class RandomForest:
    def __init__(self, n_trees=5, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, x_test):
        votes = []
        for tree in self.trees:
            pred = tree.predict([x_test])[0]
            votes.append(pred)
        prediction = max(set(votes), key=votes.count)
        return prediction

# Train and return random forest
def train_and_return_rf():
    X_train, X_test, y_train, y_test = preprocess_data("../data/ai_art", "../data/human_art")
    model = RandomForest(n_trees=5, max_depth=5)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    model = train_and_return_rf()
    print("Random Forest model trained and ready.")
