import pandas as pd
from sklearn.inspection import permutation_importance
import joblib

DATA_PATH = "data/bias_eval/test/2nd_sem_test.csv"
df = pd.read_csv(DATA_PATH, sep = ";")

cols_to_drop = [
    "gradient_boosting_pred",
    "gradient_boosting_prob",
    "logistic_regression_pred",
    "logistic_regression_prob",
    "mlp_pred",
    "mlp_prob",
    "random_forest_pred",
    "random_forest_prob",
    "Nacionality",
    "Mother_Occupation",
    "Father_Occupation"
]

df = df.drop(columns= cols_to_drop, errors="ignore")

MODEL_DIR = "outputs/models/Sec_wave/2nd_sem/"
model_type = 'random_forest'

model_path = f"{MODEL_DIR}{model_type}_best_model.pkl"
pipeline = joblib.load(model_path)

preprocess = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps['model']

feature_names = preprocess.get_feature_names_out()
importances = model.feature_importances_

fi = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print(fi.head(20))
