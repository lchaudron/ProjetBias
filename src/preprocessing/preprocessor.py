import pandas as pd
import numpy as np
import os

class DataPreprocessor:
    def __init__(self):
        # Définition des ordres logiques en tant qu'attributs de classe
        self.education_order = {
            "Basic_Education": 1, "Secondary": 2, "Technical/Specialized": 3,
            "Undergraduate": 4, "Postgraduate": 5
        }
        self.parental_education_order = {
            "Low/No_schooling": 1, "Basic_Education": 2, "Secondary": 3,
            "Technical/Specialized": 4, "Undergraduate": 5, "Postgraduate": 6
        }
        self.cat_cols = [
            'Marital status', 'Nacionality', "Mother's occupation", 
            "Father's occupation", "Application mode", "Course"
        ]

    def process_and_save(self, df_to_encode_path, ref_df_path):
        """Fonction principale pour traiter et sauvegarder les datasets."""
        
        # Chargement
        df = pd.read_csv(df_to_encode_path, sep=';')
        df_ref = pd.read_csv(ref_df_path, sep=';')

        # Création du dossier de sortie s'il n'existe pas
        

        # 1. Encodage Ordinal
        df = self._apply_ordinal_encoding(df)

        # 2. Inversion de l'ordre d'application
        df['Application order'] = 9 - df['Application order']

        # 3. Encodage One-Hot (Dummies)
        df = pd.get_dummies(df, columns=self.cat_cols, drop_first=True)

        # 4. Alignement des colonnes (df_mean doit copier la structure de df)
        cols_entrainement = df_ref.drop(columns=['Target']).columns

        for col in cols_entrainement:
            if col not in df.columns:
                df[col] = False
        df = df[cols_entrainement]

        # 5. Nettoyage et Splits temporels
        df = df.dropna()
        df_inscription = df.filter(regex='^(?!.*sem).*$')
        cols_1st_sem = [c for c in df.columns if '1st sem' in c]
        df_1st_sem = pd.concat([df_inscription, df[cols_1st_sem]], axis=1)
        df_2nd_sem = df.copy()

        return df_inscription, df_1st_sem, df_2nd_sem

    def _apply_ordinal_encoding(self, dataframe):
        """Méthode interne pour gérer le mapping des diplômes."""
        d = dataframe.copy()
        mappings = {
            'Previous qualification': self.education_order,
            "Mother's qualification": self.parental_education_order,
            "Father's qualification": self.parental_education_order
        }
        for col, mapping in mappings.items():
            if col in d.columns:
                d[f'{col}_encoded'] = d[col].map(mapping)
                d = d.drop(columns=[col])
        return d