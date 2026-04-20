from src.preprocessing.preprocessor import DataPreprocessor
import pandas as pd


preprocessor = DataPreprocessor()


path = "data/mean_student/gend-nat/Gender-0_Nacionality-Africa.csv"
path_ref = "data/final/data_2nd_sem.csv"

df_ref = pd.read_csv(path_ref, sep=';')

df_inscription, df_1st_sem, df_2nd_sem = preprocessor.process_and_save(path, path_ref)
print("Colonnes du dataset d'entraînement (sans Target) :")
print(df_ref.drop(columns=['Target']).columns)
print("\nColonnes du dataset traité :")
print(df_2nd_sem.columns)

assert set(df_ref.drop(columns=['Target']).columns) == set(df_2nd_sem.columns), "Les colonnes ne correspondent pas !"
print("\nTest réussi : Les colonnes correspondent parfaitement !")






