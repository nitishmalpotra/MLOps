import yaml
from pipeline import data_ingestion, data_preprocessing, model_training, model_evaluation

def main():
    """Main function to run the ML pipeline."""
    with open('src/config/main.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Run pipeline steps
    raw_data = data_ingestion.load_data(config['data']['raw_path'])
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess_data(raw_data, config)
    model_training.train_model(X_train, y_train, config)
    model_evaluation.evaluate_model(X_test, y_test, config)

if __name__ == "__main__":
    main()