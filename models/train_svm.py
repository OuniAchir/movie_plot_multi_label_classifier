from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def train_svm(train_df, selected_tags):
    """
    Cette fonction entraîne des modèles SVM linéaires pour chaque étiquette spécifiée
    et retourne les modèles entraînés ainsi que les résultats d'évaluation accuracy

    Arguments
    - train_df : DataFrame Spark contenant les données d'entraînement
    - selected_tags : Liste des étiquettes (tags) pour lesquelles les modèles seront entraînés

    Retourne
    - model_svm : Dictionnaire contenant les modèles Linear SVM pour chaque étiquette
    - results_svm : Dictionnaire contenant les métriques d'évaluation accuracy pour chaque étiquette
    """
    
    model_svm = {}
    results_svm = {}

    for tag in selected_tags:
        print(f"Training Linear SVM model for tag: {tag}")
        lsvc = LinearSVC(featuresCol='features', labelCol=tag, maxIter=10, regParam=0.1)
        lsvc_model = lsvc.fit(train_df)
        model_svm[tag] = lsvc_model

        predictions = lsvc_model.transform(train_df)
        evaluator = MulticlassClassificationEvaluator(labelCol=tag, predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        results_svm[tag] = accuracy
    return model_svm, results_svm
