import pandas as pd
import numpy as np

df_train = pd.read_csv("data/bias_eval/train/2nd_sem_train.csv", sep=";")
df_test = pd.read_csv("data/bias_eval/test/2nd_sem_test.csv", sep=";")

#df_train = df_train[df_train['Target'] == 0]
#df_test = df_test[df_test['Target'] == 0]


def compare_correlations(df_train, df_test, feature='Tuition fees up to date', target_metrics=['Curricular units 1st sem (grade)', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)']):
    """
    Compare la corrélation entre une variable de biais (Debtor) 
    et les variables de performance académique.
    """
    results = []
    
    for name, df in [('Train', df_train), ('Test', df_test)]:
        for metric in target_metrics:
            if metric in df.columns:
                # Calcul de la corrélation de Point-Biserial (puisque Debtor est binaire)
                corr = df[feature].corr(df[metric])
                results.append({
                    'Dataset': name,
                    'Metric': metric,
                    'Correlation_with_Debtor': corr
                })
    
    df_corr = pd.DataFrame(results)
    return df_corr.pivot(index='Metric', columns='Dataset', values='Correlation_with_Debtor')

# Utilisation
#corr_comparison = compare_correlations(df_train, df_test)
#print(corr_comparison)

import pandas as pd
import numpy as np

def analyze_joint_financial_impact(df_train, df_test):
    results = []
    target_metrics = [
        'Curricular units 1st sem (grade)', 
        'Curricular units 1st sem (approved)',
        'Curricular units 2nd sem (approved)', 
        'Curricular units 2nd sem (grade)'
    ]

    for name, df in [('Train', df_train), ('Test', df_test)]:
        # Création de la variable combinée : 
        # On veut que 1 = Stress financier élevé (Debtor ET frais non payés)
        # On inverse Tuition fees (1 - val) pour que 1 signifie "Pas à jour"
        temp_df = df.copy()
        temp_df['Financial_Stress'] = (
            (temp_df['Debtor'] == 1) & 
            (temp_df['Tuition fees up to date'] == 0)
        ).astype(int)

        for metric in target_metrics:
            if metric in temp_df.columns:
                # Corrélation entre le cumul des dettes et les notes
                corr = temp_df['Financial_Stress'].corr(temp_df[metric])
                results.append({
                    'Dataset': name,
                    'Metric': metric,
                    'Joint_Financial_Corr': corr
                })
    
    df_res = pd.DataFrame(results).pivot(index='Metric', columns='Dataset', values='Joint_Financial_Corr')
    return df_res

# --- Exécution ---
# On recharge pour être sûr des types
df_train = pd.read_csv("data/bias_eval/train/2nd_sem_train.csv", sep=";")
df_test = pd.read_csv("data/bias_eval/test/2nd_sem_test.csv", sep=";")

joint_corr = analyze_joint_financial_impact(df_train, df_test)
print("--- Corrélation : Stress Financier vs Notes ---")
print(joint_corr)