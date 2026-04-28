import pandas as pd
import re

def get_bias_report_table(df):
    df_table = df.copy()

    # 1. Nettoyage profond des chaînes de caractères (retrait des [], '', etc.)
    def clean_text(text):
        return re.sub(r"[\[\]'\"']", "", str(text)).strip()

    df_table['Variable'] = df_table['Variable'].apply(clean_text)
    df_table['Groupe'] = df_table['Groupe'].apply(clean_text)

    # 2. Dictionnaires de mapping
    binary_map = {"0": "No", "1": "Yes", "0.0": "No", "1.0": "Yes"}
    gender_map = {"0": "Female", "1": "Male", "0.0": "Female", "1.0": "Male"}
    qualif_map = {
        "2.0": "Basic_Education", "3.0": "Secondary", 
        "4.0": "Technical/Specialized", "5.0": "Undergraduate",
        "6.0": "Postgraduate"
    }

    # 3. Traduction des modalités (Valeurs)
    def map_modality(row):
        var, val = row['Variable'], row['Groupe']
        if var == 'Gender':
            return gender_map.get(val, val)
        elif 'qualification' in var:
            return qualif_map.get(val, f"Level {val}")
        elif var in ['Debtor', 'Scholarship holder', 'Displaced', 'Educational special needs', 'Tuition fees up to date']:
            return binary_map.get(val, val)
        return val

    df_table['Modality'] = df_table.apply(map_modality, axis=1)

    # 4. Sélection et nettoyage des colonnes
    model_cols = [c for c in df_table.columns if 'FPR_' in c]
    
    # Filtrage sur le Share (N > 10%)
    df_filtered = df_table[df_table['Share'] > 0.10].copy()
    
    # Construction du tableau final
    final_cols = ['Variable', 'Modality', 'Share'] + model_cols
    df_final = df_filtered[final_cols].copy()
    
    # Renommer les modèles pour le texte (FPR_mlp -> MLP)
    new_names = {c: c.replace('FPR_', '').replace('_', ' ').title() for c in model_cols}
    df_final = df_final.rename(columns=new_names)
    
    # Tri pour regrouper les variables ensemble
    df_final = df_final.sort_values(['Variable', 'Modality'])

    return df_final

# --- Exécution ---
df_final_pivot = pd.read_csv("outputs/BIAS_eval/fpr_rates/fpr_2nd_sem_test.csv")
report_table = get_bias_report_table(df_final_pivot)

# Affichage formaté
print("="*100)
print(f"{'AUDIT DES BIAIS (FPR) - SHARE > 10%':^100}")
print("="*100)
print(report_table.to_string(index=False))
print("="*100)

# Affichage avec style pour mettre en évidence les valeurs élevées (FPR > 0.15 par exemple)
print("--- TABLEAU RÉCAPITULATIF DES FPR (SHARE > 10%) ---")
#print(report_table.to_string())
report_table.to_csv("outputs/BIAS_eval/heatmap_table.csv", index=False)
# Si tu es dans un notebook Jupyter, utilise ceci pour un meilleur rendu :
# summary_table.style.background_gradient(cmap='YlOrRd', axis=None)