from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def train_naive_bayes(train_df, selected_tags):
    """
    Cette fonction entraîne des modèles de Naive Bayes pour chaque étiquette spécifiée
    et retourne les modèles entraînés ainsi que les résultats d'évaluation (accuracy).

    Arguments :
    - train_df : DataFrame Spark contenant les données d'entraînement.
    - selected_tags : Liste des étiquettes (tags) pour lesquelles les modèles seront entraînés.

    Retourne :
    - model_nb : Dictionnaire contenant les modèles Naive Bayes pour chaque étiquette.
    - results_nb : Dictionnaire contenant les métriques d'évaluation (accuracy) pour chaque étiquette.
    """
    
    model_nb = {}
    results_nb = {}

    for tag in selected_tags:
        print(f"Training Naive Bayes model for tag: {tag}")
        nb = NaiveBayes(featuresCol='features', labelCol=tag, smoothing=1.0, modelType="multinomial")
        nb_model = nb.fit(train_df)
        model_nb[tag] = nb_model

        predictions = nb_model.transform(train_df)
        evaluator = MulticlassClassificationEvaluator(labelCol=tag, predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        results_nb[tag] = accuracy
    return model_nb, results_nb
