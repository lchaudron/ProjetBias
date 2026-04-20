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


#Mean profile 
df_mean = pd.read_csv('data/mean_student/mean_student_encoded.csv', sep=';')
mean_student = df_mean.iloc[0].to_dict() 

# Variations pour créer 5 profils différents
variations = [
    {'Age at enrollment': 18, 'Admission grade': 140.0, 'Scholarship holder': 1},
    {'Age at enrollment': 25, 'Admission grade': 110.0, 'Debtor': 1},
    {'Gender': 0, 'Age at enrollment': 19, 'Admission grade': 135.0},
    {'Course': 33, 'Application order': 1, 'Tuition fees up to date': 1},
    {'Unemployment rate': 15.5, 'GDP': -0.5, 'Age at enrollment': 22}
]

test_pairs = []

for i, var in enumerate(variations):
    # Création du profil de base pour cette variation
    student_profile = mean_student.copy()
    student_profile.update(var)
    
    # 1. Version Portugaise
    portuguese_version = student_profile.copy()
    portuguese_version['Nacionality_Portugese'] = True
    portuguese_version['Nacionality_Europe'] = False
    portuguese_version['Nacionality_Asia'] = False
    portuguese_version['Nacionality_South_America'] = False
    
    # 2. Version Africaine (Toutes les colonnes Nationality à False)
    african_version = student_profile.copy()
    african_version['Nacionality_Portugese'] = False
    african_version['Nacionality_Europe'] = False
    african_version['Nacionality_Asia'] = False
    african_version['Nacionality_South_America'] = False
    african_version['Educational special needs'] = 1
    african_version['Debtor'] = 1
    african_version['Scholarship holder'] = 1
    test_pairs.append((portuguese_version, african_version))

for model in model_types:
    print(f"--- Modèle: {model} ---")
    for i, (pt, af) in enumerate(test_pairs):
        pred_pt, conf_pt, prob_pt = predire_etudiant(pt, 'Inscription', model)
        pred_af, conf_af, prob_af = predire_etudiant(af, 'Inscription', model)
        print(f"Paire {i+1} - Version Portugaise: {pred_pt} (Confiance: {conf_pt:.2f}) | Version Africaine: {pred_af} (Confiance: {conf_af:.2f})")

# --- EXEMPLE D'UTILISATION POUR PRÉDICTION ---
# for i, (pt, af) in enumerate(test_pairs):
#     df_pt = pd.DataFrame([pt])
#     df_af = pd.DataFrame([af])
#     pred_pt = model.predict_proba(df_pt)
#     pred_af = model.predict_proba(df_af)
#     print(f"Paire {i+1} - Différence de probabilité : {pred_pt - pred_af}")