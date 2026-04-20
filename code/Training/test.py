import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

file_inscription = 'data/final/data_inscription.csv'
file_1st_sem = 'data/final/data_1st_sem.csv'
file_2nd_sem = 'data/final/data_2nd_sem.csv'

df_inscription = pd.read_csv(file_inscription, sep=';')
df_1st_sem = pd.read_csv(file_1st_sem, sep=';')
df_2nd_sem = pd.read_csv(file_2nd_sem, sep=';')

list_dfs = [df_inscription, df_1st_sem, df_2nd_sem]

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}
for name, model in models.items():
    print(f'--- {name} ---')
    for df in list_dfs:
        target = 'Target'

        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(target, axis=1), df[target], test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        print(cm)