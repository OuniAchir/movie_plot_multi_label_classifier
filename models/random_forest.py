from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def train_random_forest(train_df, selected_tags):
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
