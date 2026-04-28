import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df_test = pd.read_csv('data/bias_eval/test/inscription_test.csv', sep=';')

liste_socio_dem = ['Nacionality', 'Mother_Occupation', 'Father_Occupation', 'Debtor', 'Scholarship holder', 'Gender', 'Displaced',
                   'Educational special needs', 'Tuition fees up to date', "Mother's qualification_encoded",
                   "Father's qualification_encoded"]

models = ['gradient_boosting', 'logistic_regression', 'mlp', 'random_forest']

# Liste des colonnes binaires à traduire
cols_to_map_binary = [
    'Debtor', 'Scholarship holder', 'Gender', 'Displaced', 
    'Educational special needs', 'Tuition fees up to date'
]

# Charger le dataset
df_test = pd.read_csv('data/bias_eval/test/inscription_test.csv', sep=';')

# Mapping pour les variables binaires
binary_mapping = {0: 'No', 1: 'Yes'}

# Mapping simplifié pour les qualifications (pour éviter d'avoir 30 légendes)
qualif_mapping = {
    1.0: "Low/No_schooling",
    2.0: "Basic_Education",
    3.0: "Secondary",
    4.0: "Technical/Specialized",
    5.0: "Undergraduate",
    6.0: "Postgraduate",
}

# 1. Appliquer le mapping binaire (avec une précaution pour Gender : 0=Female, 1=Male)
for col in cols_to_map_binary:
    if col in df_test.columns:
        if col == 'Gender':
            df_test[col] = df_test[col].map({0: 'Female', 1: 'Male'})
        else:
            df_test[col] = df_test[col].map(binary_mapping)

# 2. Appliquer le mapping des qualifications
qual_cols = ["Mother's qualification_encoded", "Father's qualification_encoded"]
for col in qual_cols:
    if col in df_test.columns:
        df_test[col] = df_test[col].map(qualif_mapping).fillna('Other/Unknown')

# Ensuite, tu lances tes fonctions de plot ou de calcul de FPR
# Les légendes afficheront désormais "Yes/No", "Male/Female" ou "Higher Ed"

def plot_model_disparity_grid(df, group_col, path):
    # 1. Transformer le DataFrame du format "large" au format "long" pour Seaborn
    # On passe de [Prob_RF, Prob_GB, ...] à [Model, Prob]
    model_cols = [f'{m}_prob' for m in models] # Assurez-vous que vos colonnes suivent ce nommage
    df = df[df['Target'] == 0]
    df_long = df.melt(
        id_vars=[group_col, 'Target'], 
        value_vars=model_cols,
        var_name='Model', 
        value_name='Prob_Dropout'
    )
    
    # Nettoyer le nom des modèles pour l'affichage (ex: prob_random_forest -> Random Forest)
    df_long['Model'] = df_long['Model'].str.replace('_prob', '').str.replace('_', ' ').str.title()

    # 2. Création de la grille (2 colonnes par ligne par exemple)
    g = sns.FacetGrid(df_long, col="Model", hue=group_col, col_wrap=2, height=4, aspect=1.5, palette='viridis')
    
    # 3. Ajouter les densités
    g.map(sns.kdeplot, 'Prob_Dropout', fill=True, alpha=0.4, common_norm=False, bw_adjust = 0.3)
    
    # 4. Personnalisation de chaque sous-graphique
    for ax in g.axes.flatten():
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Seuil 0.5')
        ax.set_xlabel('Dropout probability')
        ax.set_ylabel('Density')
        
    # 5. Ajouter une légende globale unique
    g.add_legend(title=f"Group : {group_col}")
    
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Comparison of density disparities for dropout prediction between models (Group : {group_col})', fontsize=16)
    
    plt.savefig(path)

# Utilisation
for feature in liste_socio_dem:
    plot_model_disparity_grid(df_test, feature, f'outputs/viz/bias_eval/auc/inscription/{feature}.png')