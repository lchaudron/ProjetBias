import pandas as pd
import numpy as np

# Supposons que ta liste soit :
# liste_df = [df_inscription, df_1st_sem, df_2nd_sem]

def reconstruct_categorical_columns(df):
    """Reconstruit les colonnes Nacionality, Mother_Occupation et Father_Occupation."""
    
    # Copie pour éviter les warnings SettingWithCopy
    df = df.copy()

    # --- 1. Reconstruction Nacionality ---
    cols_nacionality = [
        'Nacionality_Asia', 
        'Nacionality_Europe', 
        'Nacionality_Portugese', 
        'Nacionality_South_America'
    ]
    
    # On ne traite que si ces colonnes existent dans le df actuel
    present_nacionality = [c for c in cols_nacionality if c in df.columns]
    if present_nacionality:
        has_nationality = df[present_nacionality].sum(axis=1) > 0
        df['Nacionality'] = np.where(
            has_nationality, 
            df[present_nacionality].idxmax(axis=1).str.replace('Nacionality_', ''), 
            'Africa'
        )

    # --- 2. Reconstruction Occupations ---
    for parent in ['Mother', 'Father']:
        # Récupération dynamique des colonnes
        parent_cols = df.filter(like=f"{parent}'s occupation_").columns
        
        if len(parent_cols) > 0:
            # Reconstruction via idxmax
            df[f"{parent}_Occupation"] = (
                df[parent_cols].idxmax(axis=1)
                .str.replace(f"{parent}'s occupation_", "")
            )
            
            # Gestion de la catégorie de référence (si tout est à 0)
            mask_ref = df[parent_cols].sum(axis=1) == 0
            df.loc[mask_ref, f"{parent}_Occupation"] = "Administrative"
            
    return df

# --- Application à ta liste ---
# liste_df_reconstruit = [reconstruct_categorical_columns(df) for df in liste_df]

# Si tu travailles avec tes fichiers directement :
PHASES = ['inscription', '1st_sem', '2nd_sem']
for phase in PHASES:
    path = f'data/bias_eval/test/{phase}_test.csv'
    df = pd.read_csv(path, sep=';')
    
    # Appliquer la reconstruction
    df = reconstruct_categorical_columns(df)
    
    # Sauvegarder ou utiliser le df
    print(f"--- {phase.upper()} ---")
    print(df['Nacionality'].value_counts().head(2))
    df.to_csv(path, sep=';', index=False)