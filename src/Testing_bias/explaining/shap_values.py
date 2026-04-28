import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_TYPES = ['gradient_boosting', 'logistic_regression', 'mlp', 'random_forest']
DATA_PATH = "data/bias_eval/test/2nd_sem_test.csv"
MODEL_DIR = "outputs/models/Sec_wave/2nd_sem/"

# --- LOAD DATA ---
df = pd.read_csv(DATA_PATH, sep=';')

cols_to_drop = [
    "gradient_boosting_pred",
    "gradient_boosting_prob",
    "logistic_regression_pred",
    "logistic_regression_prob",
    "mlp_pred",
    "mlp_prob",
    "random_forest_pred",
    "random_forest_prob",
    "Nacionality",
    "Mother_Occupation",
    "Father_Occupation"
]

df = df.drop(columns= cols_to_drop, errors="ignore")

X = df.drop(columns=['Target'])
y = df['Target']

# --- BOUCLE MODELES ---
for model_type in MODEL_TYPES:
    print(f"\n===== {model_type.upper()} =====")
    
    model_path = f"{MODEL_DIR}{model_type}_best_model.pkl"
    
    if not os.path.exists(model_path):
        print("Model not found")
        continue
    
    model = joblib.load(model_path)
    
    # --- SHAP EXPLAINER ---
    def predict_fn(X):
        return model.predict_proba(X)

    explainer = shap.Explainer(predict_fn, X.sample(100))
    shap_values = explainer(X)
    
    # =========================
    # 1. GLOBAL IMPORTANCE
    # =========================
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Summary - {model_type}")
    plt.savefig(f"outputs/shap/{model_type}_summary.png")
    plt.close()
    
    # =========================
    # 2. FOCUS DEBTOR (BIAIS)
    # =========================
    if 'Debtor' in X.columns:
        plt.figure()
        shap.summary_plot(
            shap_values[df['Debtor'] == 1],
            X[df['Debtor'] == 1],
            show=False
        )
        plt.title(f"SHAP Debtor=1 - {model_type}")
        plt.savefig(f"outputs/shap/{model_type}_debtor1.png")
        plt.close()
    
    # =========================
    # 3. FAUX POSITIFS CRITIQUES
    # =========================
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)
    
    df['proba'] = probs
    
    fp_high = df[
        (df['Target'] == 0) &
        (df['proba'] > 0.8)
    ]
    
    if len(fp_high) > 0:
        print(f"High-confidence FP: {len(fp_high)}")
        
        X_fp = fp_high.drop(columns=['Target', 'proba'])
        
        if model_type in ['random_forest', 'gradient_boosting']:
            shap_fp = explainer.shap_values(X_fp)
            if isinstance(shap_fp, list):
                shap_fp = shap_fp[1]
        else:
            shap_fp = explainer(X_fp).values
        
        plt.figure()
        shap.summary_plot(shap_fp, X_fp, show=False)
        plt.title(f"SHAP False Positives (>0.8) - {model_type}")
        plt.savefig(f"outputs/shap/{model_type}_fp_high.png")
        plt.close()
    
    else:
        print("No high-confidence FP")