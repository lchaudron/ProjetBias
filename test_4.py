import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_test = pd.read_csv("data/bias_eval/test/2nd_sem_test.csv", sep = ";")

def plot_fpr_decay(df, feature, model_name):
    thresholds = np.linspace(0.05, 0.95, 20)
    prob_col = f'{model_name}_prob'
    
    plt.figure(figsize=(10, 6))
    
    # On ne travaille que sur les Target == 0 (pour le FPR)
    df_neg = df[df['Target'] == 0].copy()
    
    groups = df_neg[feature].unique()
    
    for group in groups:
        group_data = df_neg[df_neg[feature] == group]
        fpr_list = []
        
        for t in thresholds:
            fp = (group_data[prob_col] >= t).sum()
            tn = (group_data[prob_col] < t).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_list.append(fpr)
            
        # Mapping simple pour la légende
        label = "Yes" if group == 1 else "No" if group == 0 else group
        plt.plot(thresholds, fpr_list, marker='o', label=f'{feature}: {label}', linewidth=2)

    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Seuil standard (0.5)')
    plt.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Ton seuil (0.8)')
    
    plt.title(f'FPR Decay : Sensibilité du biais au seuil ({model_name})')
    plt.xlabel('Seuil de décision (Threshold)')
    plt.ylabel('Taux de Faux Positifs (FPR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.02, 1.02)
    plt.show()

# Exemple d'utilisation
plot_fpr_decay(df_test, 'Debtor', 'mlp')
# plot_fpr_decay(df_test, 'Gender', 'mlp')