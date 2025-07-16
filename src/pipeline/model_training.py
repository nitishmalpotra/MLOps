import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pickle

def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: dict):
    """Trains the model and logs with MLflow."""
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config['model']['params'])

        # Train model
        model = LogisticRegression(**config['model']['params'])
        model.fit(X_train, y_train)

        # Log metrics (can be done in evaluation step)
        # For simplicity, let's log a dummy metric here
        accuracy = model.score(X_train, y_train)
        mlflow.log_metric("training_accuracy", accuracy)

        # Log the model
        model_path = Path("models")
        model_path.mkdir(exist_ok=True)
        pickle.dump(model, open(model_path / "model.pkl", "wb"))
        
        mlflow.sklearn.log_model(model, "model")

        print("Model trained and logged to MLflow.")