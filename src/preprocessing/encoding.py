import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/data_mapped.csv', sep=';')
df_mean = pd.read_csv('data/mean_student/mean_student.csv', sep=';')
# Définir l'ordre logique
education_order = {
    "Basic_Education": 1,
    "Secondary": 2,
    "Technical/Specialized": 3,
    "Undergraduate": 4,
    "Postgraduate": 5
}

parental_education_order = {
    "Low/No_schooling": 1,
    "Basic_Education": 2,
    "Secondary": 3,
    "Technical/Specialized": 4,
    "Undergraduate": 5,
    "Postgraduate": 6
}

# Application
df['Previous qualification_encoded'] = df['Previous qualification'].map(education_order)
df['Mother\'s qualification_encoded'] = df['Mother\'s qualification'].map(parental_education_order)
df['Father\'s qualification_encoded'] = df['Father\'s qualification'].map(parental_education_order)
df_mean['Previous qualification_encoded'] = df_mean['Previous qualification'].map(education_order)
df_mean['Mother\'s qualification_encoded'] = df_mean['Mother\'s qualification'].map(parental_education_order)
df_mean['Father\'s qualification_encoded'] = df_mean['Father\'s qualification'].map(parental_education_order)

df = df.drop(columns=['Previous qualification', "Mother's qualification", "Father's qualification"])
df_mean = df_mean.drop(columns=['Previous qualification', "Mother's qualification", "Father's qualification"])


df['Application order'] = 9 - df['Application order']
df_mean['Application order'] = 9 - df_mean['Application order']

df = pd.get_dummies(df, columns=['Marital status', 'Nacionality', "Mother's occupation", "Father's occupation", 
                                 "Application mode", "Course"], drop_first=True)
df_mean = pd.get_dummies(df_mean, columns=['Marital status', 'Nacionality', "Mother's occupation", "Father's occupation", 
                                 "Application mode", "Course"], drop_first=True)


# On récupère la liste des colonnes du dataset d'entraînement (sans la Target)
cols_entrainement = df.drop(columns=['Target']).columns

# On ajoute les colonnes manquantes dans df_mean avec la valeur False
for col in cols_entrainement:
    if col not in df_mean.columns:
        df_mean[col] = False # Ou 0 selon ton format

# On s'assure que l'ordre des colonnes est strictement le même
# et on ne garde que les colonnes qui servent au modèle (on vire les surplus éventuels)
df_mean = df_mean[cols_entrainement]

df = df.dropna()

df_inscription = df.filter(regex='^(?!.*sem).*$')

# 2. Dataset 1er Semestre (Inscription + variables "1st sem")
cols_1st_sem = [col for col in df.columns if '1st sem' in col]
df_1st_sem = pd.concat([df_inscription, df[cols_1st_sem]], axis=1)

# 3. Dataset 2ème Semestre (Tout le dataset)
df_2nd_sem = df.copy()

#df_inscription.to_csv('data/final/data_inscription.csv', index=False, sep=';')
#df_1st_sem.to_csv('data/final/data_1st_sem.csv', index=False, sep=';')
#df_2nd_sem.to_csv('data/final/data_2nd_sem.csv', index=False, sep=';')
#df_mean.to_csv('data/mean_student/mean_student_encoded.csv', index=False, sep=';')