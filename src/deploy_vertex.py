import os, time
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID  = os.environ["GOOGLE_CLOUD_PROJECT"]
REGION      = os.getenv("VERTEX_REGION", "us-central1")
BUCKET_NAME = os.environ["MODEL_BUCKET"]
MODEL_PATH  = f"gs://{BUCKET_NAME}/models/recommendation_model/knn_recommendation_model.joblib"
MODEL_NAME  = "knn-recommender"
ENDPOINT_ID = "knn-recommender-pvt"

aiplatform.init(project=PROJECT_ID, location=REGION)

def import_model():
    model = aiplatform.Model.upload(
        display_name = MODEL_NAME,
        artifact_uri = MODEL_PATH,
        serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-4:latest",
    )
    model.wait()
    return model

def get_or_create_endpoint():
    eps = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_ID}"')
    if eps:
        return eps[0]
    return aiplatform.Endpoint.create(
        display_name = ENDPOINT_ID,
        network      = os.environ["VPC_NETWORK_URI"],  # e.g. "projects/.../global/networks/default"
        enable_private_service_connect=True,
    )

def deploy(model, endpoint):
    model.deploy(
        endpoint               = endpoint,
        machine_type           = "n1-standard-2",
        traffic_percentage     = 100,
        min_replica_count      = 1,
        max_replica_count      = 1,
        sync=True,
    )

def sanity_check(endpoint):
    resp = endpoint.predict(instances=[{"feature_1":1.5,"feature_2":4.0}])
    print("✅ Prediction response:", resp)

if __name__ == "__main__":
    model    = import_model()
    endpoint = get_or_create_endpoint()
    deploy(model, endpoint)
    sanity_check(endpoint)
