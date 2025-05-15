import os
import random
import numpy as np
from PIL import Image

from preprocessing import preprocess_data
from model_knn import KNN
from model_rf import RandomForest
from model_svm import SVM

def pick_random_image(folder, extensions=('.jpg', '.jpeg', '.png')):
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(extensions)]
    if not images:
        raise FileNotFoundError("No images found in the test folder.")
    return random.choice(images)

# Convert image to model input format
def prepare_image(image_path, size=(64, 64)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    return np.array(img).astype(np.float32).reshape(-1) / 255.0

# Evaluate test accuracy
def calculate_accuracy(model, X_test, y_test):
    predictions = [model.predict(x) for x in X_test]
    return (np.mean(np.array(predictions) == y_test)) * 100

# Print model result
def print_prediction(model, name, X_test, y_test, test_image):
    prediction = model.predict(test_image)
    label = "AI Art" if prediction == 1 else "Human Art"
    accuracy = calculate_accuracy(model, X_test, y_test)
    print(f"{name:<25}: {label:<10} | Accuracy: {accuracy:.2f}%")

# Main
if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = preprocess_data("../data/ai_art", "../data/human_art")

    print("Selecting test image...")
    image_path = pick_random_image("../data/test")
    test_image = prepare_image(image_path)
    print(f"\nSelected test image: {image_path}\n")

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    print_prediction(knn, "KNN Prediction", X_test, y_test, test_image)

    rf = RandomForest(n_trees=5, max_depth=5)
    rf.fit(X_train, y_train)
    print_prediction(rf, "Random Forest Prediction", X_test, y_test, test_image)

    svm = SVM(learning_rate=0.0001, lambda_param=0.01, n_iters=500)
    svm.fit(X_train, y_train)
    print_prediction(svm, "SVM Prediction", X_test, y_test, test_image)
