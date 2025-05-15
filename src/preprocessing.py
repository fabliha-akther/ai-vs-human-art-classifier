import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, label, size=(64, 64), limit=1000):
    data = []
    for idx, filename in enumerate(os.listdir(folder)):
        if idx >= limit:
            break
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(size)
                data.append((np.array(img), label))
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    return data

def preprocess_data(ai_dir, human_dir):
    print("Loading AI-generated images...")
    ai_data = load_images_from_folder(ai_dir, label=1, limit=1000)

    print("Loading human-made images...")
    human_data = load_images_from_folder(human_dir, label=0, limit=1000)

    all_data = ai_data + human_data
    np.random.shuffle(all_data)

    X = np.array([x for x, _ in all_data])
    y = np.array([y for _, y in all_data])

    X = X / 255.0
    X = X.reshape((X.shape[0], -1))

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Run test
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("../data/ai_art", "../data/human_art")
    print("Preprocessing complete")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train distribution:", np.bincount(y_train))
