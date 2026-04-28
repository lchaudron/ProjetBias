import pandas as pd

df = pd.read_csv("data/bias_eval/test/inscription_test.csv", sep=";")
MODEL_TYPES = ['gradient_boosting', 'logistic_regression', 'mlp', 'random_forest']

for model in MODEL_TYPES:
    print(f"Probas pour {model} :")
    # Vérifiez si vos probabilités sont bien des probabilités
    print("Max prob:", df[f'{model}_prob'].max())
    print("Min prob:", df[f'{model}_prob'].min())

# Si le max est > 1, vérifiez comment vous avez extrait la probabilité :
# CORRECT : model.predict_proba(X)[:, 1] 
# INCORRECT : model.decision_function(X)

