from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def train_gbt(train_df, selected_tags):
    """
    Cette fonction entraîne des modèles de Gradient Boosted Trees (GBT) pour chaque étiquette spécifiée
    et retourne les modèles entraînés ainsi que les résultats d'évaluation (accuracy).

    Arguments :
    - train_df : DataFrame Spark contenant les données d'entraînement.
    - selected_tags : Liste des étiquettes (tags) pour lesquelles les modèles seront entraînés.

    Retourne :
    - model_gbt : Dictionnaire contenant les modèles GBT pour chaque étiquette.
    - results_gbt : Dictionnaire contenant les métriques d'évaluation (accuracy) pour chaque étiquette.
    """
    
    model_gbt = {}
    results_gbt = {}

    for tag in selected_tags:
        print(f"Training Gradient Boosted Trees model for tag: {tag}")
        gbt = GBTClassifier(featuresCol='features', labelCol=tag, maxIter=10, maxDepth=5, maxBins=32)
        gbt_model = gbt.fit(train_df)
        model_gbt[tag] = gbt_model

        predictions = gbt_model.transform(train_df)
        evaluator = MulticlassClassificationEvaluator(labelCol=tag, predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        results_gbt[tag] = accuracy
    return model_gbt, results_gbt
