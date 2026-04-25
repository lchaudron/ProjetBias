import pandas as pd
import os
import joblib

PHASES = ['Inscription', '1st_Sem', '2nd_Sem']
MODEL_TYPES = ['gradient_boosting', 'logistic_regression', 'mlp', 'random_forest']

for phase in PHASES:
    df_path = f"data/bias_eval/test/{phase.lower()}_test.csv"
    df = pd.read_csv(df_path, sep=';')
    for model_type in MODEL_TYPES:
        model_path = f"outputs/models/Sec_wave/{phase}/{model_type}_best_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)

            X = df.drop('Target', axis=1)
            y = df['Target']
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            df[f'{model_type}_pred'] = y_pred
            df[f'{model_type}_prob'] = y_prob
    
        else:
            print(f"⚠️ Modèle introuvable : {model_path}")
    
    df.to_csv(df_path, index=False, sep=';')

