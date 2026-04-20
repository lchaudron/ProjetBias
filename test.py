import pandas as pd
import joblib
import numpy as np

# 1. Charger les pipelines (exemple pour le Random Forest à chaque étape)
phases = ['Inscription', '1st_Sem', '2nd_Sem']
model_types = ['gradient_boosting', 'logistic_regression', 'mlp', 'random_forest']

# Génération automatique du dictionnaire des chemins
model_paths = {
    phase: {
        model: f"outputs/models/{phase}/{model}_best_model.pkl"
        for model in model_types
    }
    for phase in phases
}

def predire_etudiant(data_dict, phase, model_type):
    # Transformer le dictionnaire en DataFrame d'une seule ligne
    X_new = pd.DataFrame([data_dict])
    
    # Sélectionner le bon pipeline
    pipe = joblib.load(model_paths[phase][model_type])
    
    # Obtenir la prédiction et les probabilités
    classe_predite = pipe.predict(X_new)[0]
    probabilites = pipe.predict_proba(X_new)[0]
    confiance = np.max(probabilites)
    
    # Mapping des labels (à adapter selon votre LabelEncoder)
    mapping = {0: 'No dropout', 1: 'Dropout'}
    
    return mapping[classe_predite], confiance, probabilites

df = pd.read_csv('data/mean_student/mean_student_encoded.csv', sep=';')

# Exemple d'utilisation
data_dict = df.iloc[0].to_dict()  # Convertir la première ligne en dictionnaire
phase = '2nd_Sem'

for model_type in model_types:
    prediction, confidence, probabilities = predire_etudiant(data_dict, phase, model_type)
    print(f"Modèle: {model_type}")
    print(f"Prédiction: {prediction}, Confiance: {confidence:.2f}, Probabilités: {probabilities}\n")