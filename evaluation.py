def display_results(results):
    for tag, metrics in results.items():
        print(f"Tag: {tag}, Accuracy: {metrics}")
