def train_and_evaluate_svm(train_df, test_df, tags):
    from pyspark.ml.classification import LinearSVC
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    model_svm = {}
    results_svm = {}

    for tag in tags:
        print(f"Training model for tag: {tag}")
        lsvc = LinearSVC(featuresCol='features', labelCol=tag, maxIter=10, regParam=0.1)
        lsvc_model = lsvc.fit(train_df)
        model_svm[tag] = lsvc_model

        predictions = lsvc_model.transform(test_df)
        evaluator = MulticlassClassificationEvaluator(labelCol=tag, predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        results_svm[tag] = accuracy

    return model_svm, results_svm
