import pandas as pd
import os
import joblib

PHASES = ['Inscription', '1st_Sem', '2nd_Sem']
MODEL_TYPES = ['gradient_boosting', 'logistic_regression', 'mlp', 'random_forest']

# Avant la boucle des modèles, identifiez les colonnes d'origine
original_cols = None 

for phase in PHASES:
    df_path = f"data/bias_eval/test/{phase.lower()}_test.csv"
    df = pd.read_csv(df_path, sep=';')
    
    # On définit X une seule fois avec les colonnes présentes à l'ouverture
    # en excluant 'Target' et les anciennes prédictions si elles existent déjà
    cols_to_drop = ['Target'] + [c for c in df.columns if '_pred' in c or '_prob' in c]
    X_clean = df.drop(columns=cols_to_drop)

    for model_type in MODEL_TYPES:
        model_path = f"outputs/models/Sec_wave/{phase}/{model_type}_best_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)

            # Utiliser X_clean pour être sûr de ne pas avoir de colonnes parasites
            df[f'{model_type}_pred'] = model.predict(X_clean)
            df[f'{model_type}_prob'] = model.predict_proba(X_clean)[:, 1]
    
    # Sauvegarde finale une fois que tous les modèles ont ajouté leurs colonnes
    df.to_csv(df_path, index=False, sep=';')

