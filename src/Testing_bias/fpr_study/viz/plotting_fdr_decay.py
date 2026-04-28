import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_test = pd.read_csv("data/bias_eval/test/2nd_sem_test.csv", sep = ";")

def plot_combined_fpr_decay(df, model_name='random_forest', zoom=False):
    thresholds = np.linspace(0.05, 0.95, 20)
    prob_col = f'{model_name}_prob'
    
    df_neg = df[df['Target'] == 0].copy()
    
    segments = [
        {'label': 'Debtor (Yes)', 'filter': df_neg['Debtor'] == 1, 'color': '#d62728'},
        {'label': 'Tuition fees (No)', 'filter': df_neg['Tuition fees up to date'] == 0, 'color': '#ff7f0e'},
        {'label': 'Gender (Male)', 'filter': df_neg['Gender'] == 1, 'color': '#1f77b4'},
        {'label': 'Scholarship holder (No)', 'filter': df_neg['Scholarship holder'] == 0, 'color': '#9467bd'},
        {'label': 'Displaced (No)', 'filter': df_neg['Displaced'] == 0, 'color': '#2ca02c'}
    ]
    
    plt.figure(figsize=(11, 7))
    
    all_fpr_values = []  # 👉 pour ajuster le zoom en Y intelligemment
    
    for seg in segments:
        group_data = df_neg[seg['filter']]
        fpr_list = []
        
        if len(group_data) > 0:
            for t in thresholds:
                fp = (group_data[prob_col] >= t).sum()
                tn = (group_data[prob_col] < t).sum()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fpr_list.append(fpr)
            
            all_fpr_values.extend(fpr_list)
            
            plt.plot(thresholds, fpr_list, marker='o', label=seg['label'], 
                     color=seg['color'], linewidth=3, markersize=6)

    # Lignes de référence
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=0.8, color='green', linestyle=':', linewidth=2)

    # 👉 ZOOM
    if zoom:
        plt.xlim(0.5, 0.95)
        
        # zoom dynamique sur Y (évite d’avoir un graphe plat)
        if all_fpr_values:
            ymin = min(all_fpr_values)
            ymax = max(all_fpr_values)
            margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.01
            plt.ylim(max(0, ymin - margin), min(1, ymax + margin))
        
        plt.title(f'FPR Decay (Zoom > 0.5) - {model_name.upper()}', fontsize=14)
    else:
        plt.xlim(0.04, 0.96)
        plt.ylim(-0.02, 1.02)
        plt.title(f'FPR Decay - {model_name.upper()}', fontsize=14)

    plt.xlabel('Decision threshold', fontsize=12)
    plt.ylabel('False positive rate', fontsize=12)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    plt.show()

plot_combined_fpr_decay(df_test, 'random_forest', zoom=True)