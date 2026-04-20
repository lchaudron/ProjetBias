import pandas as pd
import joblib
import numpy as np
import os
import json
from src.preprocessing.preprocessor import DataPreprocessor

# 1. Configuration
ROOT_MEAN_STUDENTS = 'data/mean_student/'  # Dossier racine qui contient les sous-dossiers
PATH_REF_DATA = 'data/final/data_2nd_sem.csv'
OUTPUT_JSON = 'outputs/predictions/predictions_complet.json'

phases = ['Inscription', '1st_Sem', '2nd_Sem'] 
model_types = ['gradient_boosting', 'logistic_regression', 'mlp', 'random_forest']

# 2. Chargement des modèles
print("Chargement des modèles...")
models_pool = {}
for phase in phases:
    models_pool[phase] = {}
    for mt in model_types:
        path = f"outputs/models/{phase}/{mt}_best_model.pkl"
        if os.path.exists(path):
            models_pool[phase][mt] = joblib.load(path)
        else:
            print(f"⚠️ Modèle introuvable : {path}")

# 3. Initialisation
preprocessor = DataPreprocessor()
mapping_labels = {0: 'No dropout', 1: 'Dropout'}
results = {}

print(f"🔎 Recherche récursive des fichiers CSV dans {ROOT_MEAN_STUDENTS}...")

# --- ÉTAPE RECHERCHE RÉCURSIVE ---
# os.walk parcourt ROOT_MEAN_STUDENTS, ses sous-dossiers, etc.
for root, dirs, files in os.walk(ROOT_MEAN_STUDENTS):
    for filename in files:
        if filename.endswith('.csv') and not filename.endswith('_encoded.csv'):
            file_path = os.path.join(root, filename)
            
            # On crée une clé unique pour le JSON incluant le nom du sous-dossier
            # Exemple : "nationality/eleve_1.csv"
            relative_path = os.path.relpath(file_path, ROOT_MEAN_STUDENTS)
            results[relative_path] = {}
            
            try:
                # --- ÉTAPE A : Prétraitement ---
                df_ins, df_s1, df_s2 = preprocessor.process_and_save(
                    df_to_encode_path=file_path, 
                    ref_df_path=PATH_REF_DATA
                )
                
                data_map = {
                    'Inscription': df_ins,
                    '1st_Sem': df_s1,
                    '2nd_Sem': df_s2
                }

                # --- ÉTAPE B : Prédictions ---
                for phase in phases:
                    results[relative_path][phase] = {}
                    df_current = data_map[phase]
                    
                    if df_current.empty: continue

                    for mt in model_types:
                        if mt not in models_pool[phase]: continue
                        
                        model = models_pool[phase][mt]
                        probs = model.predict_proba(df_current)[0]
                        pred = model.predict(df_current)[0]
                        
                        results[relative_path][phase][mt] = {
                            "prediction": mapping_labels[int(pred)],
                            "prob_dropout": round(float(probs[1]), 4),
                            "confidence": round(float(np.max(probs)), 4)
                        }

            except Exception as e:
                print(f"⚠️ Erreur sur {relative_path}: {e}")

# 4. Export final
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"✅ Analyse terminée. {len(results)} profils analysés.")
print(f"📂 Résultats sauvegardés dans : {OUTPUT_JSON}")