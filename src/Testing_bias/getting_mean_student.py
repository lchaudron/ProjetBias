import pandas as pd
import os

# Chargement du dataset
df = pd.read_csv('data/processed/data_mapped.csv', sep=';')

def obtenir_profil_moyen(df, filtres=None):
    """
    Calcule le profil type d'un élève selon des filtres dynamiques.
    Renvoie un DataFrame vide si aucun élève ne correspond.
    """
    df_filtre = df.copy()
    
    # 1. Application des filtres dynamiques
    if filtres:
        for col, val in filtres.items():
            if col in df_filtre.columns:
                df_filtre = df_filtre[df_filtre[col] == val]
            else:
                print(f"Attention : La colonne '{col}' n'existe pas.")
    
    # Si le filtre ne retourne rien, on renvoie un DataFrame vide immédiatement
    if df_filtre.empty:
        return pd.DataFrame()

    profil_type = {}
    
    # 2. Calcul des statistiques par type de colonne
    for col in df_filtre.columns:
        if col == 'Target':
            continue
            
        # Médiane pour le numérique continu
        if col in ['Age at enrollment', 'Admission grade', 'Unemployment rate', 
                  'Inflation rate', 'GDP'] or "grade" in col.lower():
            profil_type[col] = df_filtre[col].median()
            
        # Médiane pour les compteurs d'unités
        elif "Curricular units" in col:
            profil_type[col] = df_filtre[col].median()

        # Mode pour le reste (catégories)
        else:
            mode_val = df_filtre[col].mode()
            profil_type[col] = mode_val[0] if not mode_val.empty else None
            
    return pd.DataFrame([profil_type])

# --- EXÉCUTION ---

# Dossier de sortie
output_dir = 'data/mean_student/gend-nat-schol-edn-debtor/'
os.makedirs(output_dir, exist_ok=True)

profil_portugais_df = obtenir_profil_moyen(df, filtres={'Nacionality': 'Portugese'})

for nat in df['Nacionality'].unique():
    for gender in df['Gender'].unique():
        val_gender = int(gender) 
    
        critères_specifiques = {
        'Gender': val_gender,
        'Scholarship holder': 1,
        'Educational special needs': 1,
        'Nacionality': nat,
        'Debtor': 1
    }
    
        eleve_type = obtenir_profil_moyen(df, filtres=critères_specifiques)
    
    # On ne sauvegarde que si le DataFrame n'est pas vide
        if not eleve_type.empty:
        # Nettoyage du nom de fichier
            filename = str(critères_specifiques).translate(str.maketrans({"{": "", "}": "", "'": "", " ": "", ",": "_", ":": "-"}))
            eleve_type.to_csv(f'{output_dir}{filename}.csv', index=False, sep=';')
        else:
            print(f"Profil inexistant pour {critères_specifiques}, création d'un profil basé sur le standard portugais.")
        # On copie le profil portugais
            eleve_type = profil_portugais_df.copy()
        
        # On injecte les valeurs forcées pour que le fichier corresponde aux critères
            for col, val in critères_specifiques.items():
                eleve_type[col] = val
            
            filename = str(critères_specifiques).translate(str.maketrans({"{": "", "}": "", "'": "", " ": "", ",": "_", ":": "-"}))
            eleve_type.to_csv(f'{output_dir}{filename}.csv', index=False, sep=';')