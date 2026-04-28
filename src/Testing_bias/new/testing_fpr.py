import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

models = ['gradient_boosting', 'logistic_regression', 'mlp', 'random_forest']
liste_socio_dem = ['Nacionality', 'Mother_Occupation', 'Father_Occupation', 'Debtor', 'Scholarship holder', 'Gender', 'Displaced',
                   'Educational special needs', 'Tuition fees up to date', "Mother's qualification_encoded",
                   "Father's qualification_encoded"]

#df_test = pd.read_csv("data/bias_eval/test/inscription_test.csv", sep=';')

def calculate_fpr_per_model(df, group_col: str):
    results = []
    groups = df[group_col].unique()
    df_size = len(df)
    for group in groups:
        group_df = df[df[group_col] == group]
        share = len(group_df)/df_size
        # Filtre sur les réels négatifs (ceux qui ne décrochent pas)
        reels_negatifs = group_df[group_df['Target'] == 0]
        
        if len(reels_negatifs) == 0:
            continue

        for model in models:
            pred_col = f'{model}_pred'
            if pred_col not in group_df.columns:
                continue
            
            # Calcul du FPR : FP / (FP + TN)
            fp = (reels_negatifs[pred_col] == 1).sum()
            tn = (reels_negatifs[pred_col] == 0).sum()
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            
            results.append({
                'Variable': group_col,
                'Groupe': group,
                'Model': model,
                'FPR': fpr,
                'Share' : share
            })

    return pd.DataFrame(results)

# --- Boucle sur la liste ---
files = ["data/bias_eval/test/inscription_test.csv",
         "data/bias_eval/test/1st_sem_test.csv",
         "data/bias_eval/test/2nd_sem_test.csv"]

for file in files : 

    df_test = pd.read_csv(file, sep = ';')
    all_results = []

    for col in liste_socio_dem:
        if col in df_test.columns:
            res_df = calculate_fpr_per_model(df_test, col)
            all_results.append(res_df)

    # Fusionner tous les résultats
    df_final_long = pd.concat(all_results, ignore_index=True)

    # Pivotage pour avoir une lecture par Variable/Groupe en ligne et Modèles en colonnes
    df_final_pivot = df_final_long.pivot(index=['Variable', 'Groupe', "Share"], columns='Model', values='FPR')

    # Nettoyage des noms de colonnes
    df_final_pivot.columns = [f'FPR_{col}' for col in df_final_pivot.columns]
    df_final_pivot = df_final_pivot.reset_index()
    df_final_pivot.to_csv(f"outputs/BIAS_eval/fpr_rates/fpr_{file.split("/")[-1]}")
