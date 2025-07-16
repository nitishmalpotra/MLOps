import mlflow
import pickle

def deploy_production_model(config: dict):
    """Fetches the production model from MLflow Registry."""
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    client = mlflow.tracking.MlflowClient()

    try:
        latest_version = client.get_latest_versions(name=config['model']['name'], stages=["Production"])[0]
        model_uri = f"runs:/{latest_version.run_id}/model"

        # Load the model
        production_model = mlflow.sklearn.load_model(model_uri)

        # "Deploy" it by saving to a known location
        with open("path/to/serve/model.pkl", "wb") as f:
            pickle.dump(production_model, f)

        print(f"Model version {latest_version.version} deployed successfully.")

    except IndexError:
        print("No model in Production stage found.")