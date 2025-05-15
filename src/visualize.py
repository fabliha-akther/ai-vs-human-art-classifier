import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

models = ['KNN', 'Random Forest', 'SVM']
accuracy = [72, 78.25, 78]
precision = [0.71, 0.76, 0.77]
recall = [0.70, 0.79, 0.78]

confusions = [
    np.array([[135, 40], [35, 90]]),
    np.array([[140, 30], [28, 100]]),
    np.array([[138, 32], [30, 98]])
]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracy, color=['skyblue', 'lightgreen', 'salmon'], edgecolor='black')
plt.ylim(60, 85)
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
for bar, acc in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width() / 2, acc + 1, f'{acc}%', ha='center')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

x = np.arange(len(models))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, precision, width, label='Precision', color='mediumpurple')
ax.bar(x + width/2, recall, width, label='Recall', color='orange')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Score")
ax.set_title("Precision and Recall")
ax.legend()
ax.set_ylim(0, 1)
for i in x:
    ax.text(i - width/2, precision[i] + 0.02, f'{precision[i]:.2f}', ha='center')
    ax.text(i + width/2, recall[i] + 0.02, f'{recall[i]:.2f}', ha='center')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, cm, name in zip(axes, confusions, models):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(f'{name} Confusion Matrix')
plt.suptitle("Confusion Matrices")
plt.tight_layout()
plt.show()
