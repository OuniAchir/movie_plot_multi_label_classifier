from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def train_random_forest(train_df, selected_tags):
    """
    Cette fonction entraîne des modèles de forêts aléatoires (Random Forest) pour chaque étiquette spécifiée
    et retourne les modèles entraînés ainsi que les résultats d'évaluation accuracy et F1-score

    Arguments 
    - train_df : DataFrame Spark contenant les données d'entraînement.
    - selected_tags : Liste des étiquettes (tags) pour lesquelles les modèles seront entraînés

    Retourne 
    - models : Dictionnaire contenant les modèles de forêts aléatoires pour chaque étiquette.
    - results : Dictionnaire contenant les métriques d'évaluation accuracy et F1-score pour chaque étiquette
    """
    
    models = {}
    results = {}
    for tag in selected_tags:
        rf = RandomForestClassifier(featuresCol='features', labelCol=tag, numTrees=10)
        model = rf.fit(train_df)
        models[tag] = model
        
        predictions = model.transform(train_df)
        evaluator = MulticlassClassificationEvaluator(labelCol=tag, predictionCol="prediction")
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        results[tag] = {"accuracy": accuracy, "f1": f1_score}
    
    return models, results
