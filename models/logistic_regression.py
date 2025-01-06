from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def train_logistic_regression(train_df, selected_tags):
    models = {}
    results = {}
    for tag in selected_tags:
        lgr = LogisticRegression(featuresCol='features', labelCol=tag)
        model = lgr.fit(train_df)
        models[tag] = model
        
        predictions = model.transform(train_df)
        evaluator = MulticlassClassificationEvaluator(labelCol=tag, predictionCol="prediction")
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        results[tag] = {"accuracy": accuracy, "f1": f1_score}
    
    return models, results
