import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

df_final_pivot = pd.read_csv("outputs/BIAS_eval/fpr_rates/fpr_2nd_sem_test.csv")

def plot_bias_heatmap_cleaned(df):
    df_plot = df.copy()

    # 1. Nettoyage profond des noms de variables et des groupes
    # On retire les crochets, les guillemets et les espaces superflus
    def clean_text(text):
        return re.sub(r"[\[\]'\"']", "", str(text)).strip()

    df_plot['Variable'] = df_plot['Variable'].apply(clean_text)
    df_plot['Groupe'] = df_plot['Groupe'].apply(clean_text)

    # 2. Application des mappings
    # On définit les dictionnaires de traduction
    binary_map = {"0": "No", "1": "Yes", "0.0": "No", "1.0": "Yes"}
    gender_map = {"0": "Female", "1": "Male", "0.0": "Female", "1.0": "Male"}
    qualif_map = {
        "2.0": "Basic_Education", "3.0": "Secondary", 
        "4.0": "Technical/Specialized", "5.0": "Undergraduate",
        "6.0": "Postgraduate"
    }

    new_labels = []
    for _, row in df_plot.iterrows():
        var = row['Variable']
        val = row['Groupe']
        
        # Logique de traduction selon la variable
        if var == 'Gender':
            readable_val = gender_map.get(val, val)
        elif 'qualification' in var:
            readable_val = qualif_map.get(val, f"Level {val}")
        elif var in ['Debtor', 'Scholarship holder', 'Displaced', 'Educational special needs', 'Tuition fees up to date']:
            readable_val = binary_map.get(val, val)
        else:
            readable_val = val
            
        new_labels.append(f"{var} ({readable_val})")

    df_plot['Final_Label'] = new_labels

    # 3. Préparation des données pour la Heatmap
    model_cols = [c for c in df_plot.columns if 'FPR_' in c]
    
    # On filtre pour ne garder que les groupes significatifs (> 5% de part)
    df_filtered = df_plot[df_plot['Share'] > 0.1].copy()
    
    # Pivotage
    heatmap_data = df_filtered.set_index('Final_Label')[model_cols]
    
    # Nettoyage des noms de colonnes (modèles)
    heatmap_data.columns = [c.replace('FPR_', '').replace('_', ' ').title() for c in heatmap_data.columns]

    # 4. Affichage
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt=".2f", linewidths=.5)
    
    plt.title('FPR rates accross models and socio-demographical features (share of N > 10%)', fontsize=15)
    plt.ylabel('Feature')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.show()

# Utilisation
plot_bias_heatmap_cleaned(df_final_pivot)