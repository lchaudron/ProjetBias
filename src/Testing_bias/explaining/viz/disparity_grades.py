import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv("data/bias_eval/train/2nd_sem_train.csv", sep=";")
df_test = pd.read_csv("data/bias_eval/test/2nd_sem_test.csv", sep=";")

df_train = df_train[df_train['Target'] == 0]
df_test = df_test[df_test['Target'] == 0]

def compare_debtor_profiles(df_train, df_test, metrics=['Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)']):
    """
    Compare le profil des Debtors entre le set d'entraînement et le set de test.
    """
    # 1. Isoler les Debtors
    train_debtors = df_train[df_train['Debtor'] == 1].copy()
    test_debtors = df_test[df_test['Debtor'] == 1].copy()
    
    train_debtors['Dataset'] = 'Train'
    test_debtors['Dataset'] = 'Test'
    
    combined = pd.concat([train_debtors, test_debtors])
    
    # 2. Création d'un tableau récapitulatif
    summary = combined.groupby('Dataset')[metrics + ['Target']].mean()
    print("--- Comparaison des moyennes (Debtors uniquement) ---")
    print(summary)
    
    # 3. Visualisation des distributions
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    
    for i, metric in enumerate(metrics):
        sns.kdeplot(data=combined, x=metric, hue='Dataset', fill=True, ax=axes[i], common_norm=False)
        axes[i].set_title(f'{metric}')
        axes[i].set_xlabel('Value')
        
    plt.tight_layout()
    plt.show()
    
    return summary

# Utilisation :
res = compare_debtor_profiles(df_train, df_test)