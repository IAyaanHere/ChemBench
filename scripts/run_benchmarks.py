import argparse
# Ab yeh imports fail nahi honge kyunki setup.py hai
from chembench.baselines import RandomForestBaseline
from chembench.datasets import DatasetLoader
from chembench.benchmarks import Evaluator

def run_benchmarks(dataset_name, model_name):
    # Load the dataset
    loader = DatasetLoader(name=dataset_name)
    X_train, X_test, y_train, y_test = loader.get_splits()

    # Initialize model
    if model_name == 'random_forest':
        model = RandomForestBaseline()
    else:
        raise ValueError(f"Model '{model_name}' is not supported yet.")

    # Train and predict
    print(f"Training {model_name} on {dataset_name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate
    evaluator = Evaluator(task_type='classification')
    results = evaluator.calculate_metrics(y_test, predictions)

    print(f"\nResults for {model_name} on {dataset_name}:")
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks for ChemBench')
    parser.add_argument('--dataset', type=str, default='tennessee_eastman', help='Dataset name')
    parser.add_argument('--model', type=str, default='random_forest', help='Model name')

    args = parser.parse_args()
    run_benchmarks(args.dataset, args.model)
