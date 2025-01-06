from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def train_naive_bayes(train_df, test_df, selected_tags):
    model_nb = {}
    results_nb = {}

    for tag in selected_tags:
        print(f"Training Naive Bayes model for tag: {tag}")
        
        # Initialiser et entraîner le modèle
        nb = NaiveBayes(featuresCol='features', labelCol=tag, smoothing=1.0, modelType="multinomial")
        nb_model = nb.fit(train_df)
        model_nb[tag] = nb_model
        
        # Prédictions sur le jeu de test
        predictions = nb_model.transform(test_df)
        
        # Évaluation
        evaluator = MulticlassClassificationEvaluator(labelCol=tag, predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        results_nb[tag] = accuracy
    
    return model_nb, results_nb
