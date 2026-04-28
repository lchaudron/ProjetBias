import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Chargement du dataset de test de la phase 2nd_Sem
df_test = pd.read_csv('data/bias_eval/test/2nd_sem_test.csv', sep=';')

models = ['gradient_boosting', 'logistic_regression', 'mlp', 'random_forest']
performance_results = []

for model in models:
    y_true = df_test['Target']
    y_pred = df_test[f'{model}_pred']
    y_prob = df_test[f'{model}_prob']
    
    performance_results.append({
        'Model': model.replace('_', ' ').title(),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob)
    })

# Création du tableau récapitulatif
df_perf = pd.DataFrame(performance_results).set_index('Model')

# Affichage formaté
print("--- MÉTRIQUES DE PERFORMANCE : PHASE 2ND SEMESTER ---")
print(df_perf.round(3).to_string())