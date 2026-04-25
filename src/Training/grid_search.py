import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# 1. Identifiez vos types de colonnes
# (Exemple à adapter selon vos noms de colonnes réels)
cols_continues = [
    'Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]



# Fichiers
files = {
    'Inscription': 'data/final/data_inscription.csv',
    '1st_Sem': 'data/final/data_1st_sem.csv',
    '2nd_Sem': 'data/final/data_2nd_sem.csv'
}

# Définition des modèles et de leurs grilles de paramètres
# Définition corrigée des modèles et de leurs grilles de paramètres
models_and_params = {
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        }
    ),
    'Logistic Regression': (
        LogisticRegression(max_iter=2000, random_state=42),
        {
            'model__C': [0.1, 1, 10],
            'model__solver': ['lbfgs', 'saga']
        }
    ),
    'MLP': (
        MLPClassifier(max_iter=1000, random_state=42),
        {
            'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [0.0001, 0.05]
        }
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(random_state=42),
        {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.1, 0.2],
            'model__max_depth': [3, 5]
        }
    ),
}

# Boucle sur les 3 moments (Datasets)
for stage_name, file_path in files.items():
    print(f"\n{'='*30}\nÉTAPES : {stage_name}\n{'='*30}")
    df = pd.read_csv(file_path, sep=';')
    
    # On définit dynamiquement les colonnes continues présentes dans ce dataframe
    # (Certaines colonnes 'sem' ne sont pas dans le dataset 'inscription')
    current_cont_cols = [c for c in cols_continues if c in df.columns]
    
    X = df.drop('Target', axis=1)
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    data_train = pd.concat([X_train, y_train], axis=1)
    data_test = pd.concat([X_test, y_test], axis=1)
    data_train.to_csv(f"data/bias_eval/train/{stage_name.lower()}_train.csv", index=False)
    data_test.to_csv(f"data/bias_eval/test/{stage_name.lower()}_test.csv", index=False)


    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), current_cont_cols)
    ], remainder='passthrough')

    # Boucle sur les modèles
    for model_name, (model, params) in models_and_params.items():
        print(f'\n--- Optimisation de {model_name} ---')
        pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
])
        
        # Grid Search avec validation croisée (cv=3 pour gagner du temps)
        grid_search = GridSearchCV(pipeline, params, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Meilleur modèle trouvé
        best_model = grid_search.best_estimator_
        print(f"Meilleurs paramètres : {grid_search.best_params_}")

        # Évaluation
        y_pred = best_model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Sauvegarde du modèle optimisé
        model_filename = f"outputs/models/Sec_wave/{stage_name}/{model_name.lower().replace(' ', '_')}_best_model.pkl"
        joblib.dump(best_model, model_filename)
        
        