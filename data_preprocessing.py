from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split

def initialize_spark(app_name="MovieML"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .getOrCreate()
    return spark

def load_and_preprocess_data(spark, file_path):
    df = spark.read.option("multiLine", "true") \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .option("ignoreLeadingWhiteSpace", True) \
        .csv(file_path, inferSchema=True, header=True)
    
    df = df.where(col('imdb_id').like("tt%"))
    # Filter rows with valid IMDb IDs
    df = df.where(col('imdb_id').like("tt%"))
    
    # Lowercase the text, remove punctuation, and special characters
    df = df.withColumn('plot_synopsis', lower(col('plot_synopsis')))
    df = df.withColumn('plot_synopsis', regexp_replace(col('plot_synopsis'), r'[^a-z\s]', ''))
    
    # Split the tags into individual rows
    df = df.withColumn('tags_split', explode(split('tags', ', ')))
    return df

def encode_data(df, selected_tags):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
    
    tokenizer = Tokenizer(inputCol='plot_synopsis', outputCol='words')
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="features")
    
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF])
    pipeline_model = pipeline.fit(df)
    encoded_df = pipeline_model.transform(df)
    
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
