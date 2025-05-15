# AI vs Human Art Classifier 🎨🤖

This project uses machine learning to distinguish between **AI-generated art** and **human-made art**. It includes three models: **K-Nearest Neighbors (KNN)**, **Random Forest**, and a **custom Support Vector Machine (SVM)**, trained on image datasets sourced from Kaggle.

---

## 📁 Project Structure

```
ai-vs-human-art-classifier/
├── data/
│   ├── ai_art/        # AI-generated images (from Kaggle)
│   ├── human_art/     # Human-created images (from Kaggle)
│   └── test/          # Images for prediction
├── src/
│   ├── preprocessing.py      # Data loading and normalization
│   ├── model_knn.py          # Custom KNN implementation
│   ├── model_rf.py           # Random Forest (sklearn-based)
│   ├── model_svm.py          # Custom linear SVM from scratch
│   ├── predict.py            # Test prediction on random image
│   └── visualize.py          # Accuracy, precision, recall plots
├── .gitignore
├── README.md
└── requirements.txt
```

> ⚠️ `data/` is excluded from Git tracking via `.gitignore`. You must manually add it after cloning.

---

## 📦 Dataset

- **Source**: [Kaggle](https://www.kaggle.com/)  
  (You must download your dataset manually. Link depends on which dataset you used.)

### Folder Setup:
Place images like this:

```
data/
├── ai_art/       # e.g., 1000 AI-generated images
├── human_art/    # e.g., 1000 human artworks
└── test/         # optional: any image for live testing
```

---

## 🚀 Running the Project

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Preprocess Dataset

```bash
python src/preprocessing.py
```

### 3. Train Models

```bash
python src/model_knn.py
python src/model_rf.py
python src/model_svm.py
```

### 4. Run Prediction on Random Test Image

```bash
python src/predict.py
```

Output:
```
Selected test image: data/test/sample.jpg

KNN Prediction            : Human Art  | Accuracy: 72.00%
Random Forest Prediction  : AI Art     | Accuracy: 78.25%
SVM Prediction            : AI Art     | Accuracy: 78.00%
```

### 5. Visualize Performance

```bash
python src/visualize.py
```

This shows:
- Accuracy bar chart
- Precision & recall bars
- Confusion matrices

---

## 🛠 Requirements

Installed via:
```bash
pip install -r requirements.txt
```

Main libraries:
- `numpy`
- `Pillow`
- `matplotlib`
- `scikit-learn`

---

## 📌 Notes

- Images are resized to **64x64 RGB**.
- Labels: `0 = Human`, `1 = AI`.
- SVM and KNN are implemented from scratch.
- Dataset size is limited to 1000 per class (adjustable in code).
- `data/` is not included in this repo due to size constraints.

---

## 🙏 Credits

- Dataset: Kaggle contributors  
- Code written in Python 3.13 using PyCharm

---

## 🔗 License

This project is free for academic use. All image rights remain with the original dataset creators.
