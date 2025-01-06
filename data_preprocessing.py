from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF

# Fonction pour initialiser une session Spark
def initialize_spark(app_name="MovieML"):
    # Création de la session Spark avec une configuration spécifique (mémoire, cœurs, etc.)
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .getOrCreate()
    return spark

def load_and_preprocess_data(spark, file_path):
    # Chargement du fichier CSV avec des options spécifiques pour gérer les lignes multi-lignes, les guillemets.. 
    df = spark.read.option("multiLine", "true") \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .option("ignoreLeadingWhiteSpace", True) \
        .csv(file_path, inferSchema=True, header=True)
    
    df = df.where(col('imdb_id').like("tt%"))
    # Filter les lignes pars imbd valide
    df = df.where(col('imdb_id').like("tt%"))
    
    # Mettre tout le texte en minuscule et supprimer la ponctuation et les caractères spéciaux
    df = df.withColumn('plot_synopsis', lower(col('plot_synopsis')))
    df = df.withColumn('plot_synopsis', regexp_replace(col('plot_synopsis'), r'[^a-z\s]', ''))
    
    # Diviser la colonne "tags" en plusieurs lignes, chaque ligne contenant un tag unique
    df = df.withColumn('tags_split', explode(split('tags', ', ')))
    return df

# Fonction pour encoder les données en utilisant Tokenizer, StopWordsRemover et HashingTF
def encode_data(df, selected_tags):    
    # Tokenization : séparer le texte en mots individuels
    tokenizer = Tokenizer(inputCol='plot_synopsis', outputCol='words')
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
        
    # Conversion des mots en vecteurs numériques
    hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="features")

    # Transformation des données d'entrée en utilisant le pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF])
    pipeline_model = pipeline.fit(df)
    encoded_df = pipeline_model.transform(df)

    # Création d'une table pivot où chaque tag devient une colonne
    pivot_df = encoded_df.groupby('plot_synopsis', 'features', 'split') \
        .pivot('tags_split', selected_tags).count().fillna(0)
    return pivot_df

def save_data(df, train_path, test_path, val_path):
    train_df = df.where(col('split') == 'train').drop('plot_synopsis', 'split')
    test_df = df.where(col('split') == 'test').drop('plot_synopsis', 'split')
    val_df = df.where(col('split') == 'val').drop('plot_synopsis', 'split')

    train_df.write.parquet(train_path)
    test_df.write.parquet(test_path)
    val_df.write.parquet(val_path)
    return train_df, test_df, val_df
