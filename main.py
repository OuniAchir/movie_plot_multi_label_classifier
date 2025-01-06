from data_preprocessing import initialize_spark, load_and_preprocess_data, encode_data, save_data
from models.train_naive_bayes import train_naive_bayes
from models.train_svm import train_svm
from models.train_gbt import train_gbt
from evaluation import display_results

# Initialisation
spark = initialize_spark()
selected_tags = ['murder', 'violence', 'flashback', 'romantic', 'cult', 'revenge', 'psychedelic', 'comedy', 'suspenseful', 'good versus evil']

# Préparation des données
df = load_and_preprocess_data(spark, "mpst_full_data.csv")
pivot_df = encode_data(df, selected_tags)
train_df, test_df, val_df = save_data(pivot_df, "train_data.parquet", "test_data.parquet", "val_data.parquet")

# Entraînement des modèles
nb_models, nb_results = train_naive_bayes(train_df, selected_tags)
svm_models, svm_results = train_svm(train_df, selected_tags)
gbt_models, gbt_results = train_gbt(train_df, selected_tags)

# Afficher les résultats
print("\nNaive Bayes Results:")
display_results(nb_results)
print("\nSVM Results:")
display_results(svm_results)
print("\nGBT Results:")
display_results(gbt_results)
