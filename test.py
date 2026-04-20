from src.preprocessing.preprocessor import Preprocessor

preprocessor = Preprocessor()


path = "data/mean_student/gend-nat/Gender-0_Nacionality-Africa.csv"
path_ref = "data/final/data_2nd_sem.csv"

df_inscription, df_1st_sem, df_2nd_sem = preprocessor.process_and_save(path, path_ref)




