import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('data/brute/data.csv', sep=';')
output_dir = os.path.join('outputs', 'viz', 'preliminary')
os.makedirs(output_dir, exist_ok=True)

target = 'Target'

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(target, axis=1), df[target], test_size=0.2, random_state=42
)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

labels = sorted(pd.unique(y_test))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'--- {name} ---')
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{name} Confusion Matrix')
    fig.tight_layout()

    filename = f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f'Saved confusion matrix plot to {os.path.join(output_dir, filename)}')
