from statsmodels.stats.proportion import proportions_ztest
import numpy as np
import pandas as pd

df_test = pd.read_csv('data/bias_eval/test/2nd_sem_test.csv', sep=';')
models = ['gradient_boosting', 'logistic_regression', 'mlp', 'random_forest']

def analyze_fpr_significance(df, group_cols, models):
    results = []
    
    for col in group_cols:
        # On s'assure que la colonne est binaire (0 et 1)
        if set(df[col].unique()) != {0, 1}:
            continue
            
        # On ne travaille que sur les individus dont la réalité est "Non-Dropout" (Target=0)
        # car le FPR se calcule sur cette population uniquement
        df_neg = df[df['Target'] == 0]
        
        for model in models:
            pred_col = f'{model}_pred'
            
            # Groupe 1 (ex: Boursiers)
            grp1 = df_neg[df_neg[col] == 1]
            fp1 = (grp1[pred_col] == 1).sum()
            n1 = len(grp1)
            
            # Groupe 0 (ex: Non-Boursiers)
            grp0 = df_neg[df_neg[col] == 0]
            fp0 = (grp0[pred_col] == 1).sum()
            n0 = len(grp0)
            
            # Calcul du Z-test (proportions_ztest)
            # count = nombre de succès (ici les erreurs FP)
            # nobs = nombre d'observations totales dans le groupe
            if n1 > 0 and n0 > 0:
                count = np.array([fp1, fp0])
                nobs = np.array([n1, n0])
                stat, p_value = proportions_ztest(count, nobs)
            else:
                p_value = np.nan
            
            results.append({
                'Caractéristique': col,
                'Modèle': model,
                'FPR_Grp_1': fp1/n1 if n1 > 0 else 0,
                'FPR_Grp_0': fp0/n0 if n0 > 0 else 0,
                'Diff_Absolue': abs((fp1/n1) - (fp0/n0)) if (n1 > 0 and n0 > 0) else 0,
                'P-Value': p_value,
                'Significatif': 'OUI' if p_value < 0.05 else 'NON'
            })
            
    return pd.DataFrame(results)

# Utilisation
colonnes_a_tester = ['Scholarship holder', 'Debtor', 'Gender', 'International']
df_significance = analyze_fpr_significance(df_test, colonnes_a_tester, models)
df_significance.to_csv('outputs/test_significance_2nd_sem.csv', sep = ";", index = False)
# Afficher uniquement les cas problématiques
print(df_significance[df_significance['Significatif'] == 'OUI'])