"""
Upload the serialized model to the configured GCS bucket.
Requires: GOOGLE_CLOUD_PROJECT and MODEL_BUCKET env vars
"""

import os, pathlib, joblib
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID   = os.environ["GOOGLE_CLOUD_PROJECT"]
BUCKET_NAME  = os.environ["MODEL_BUCKET"]         # e.g. vertexai-sdh
MODEL_FILE   = pathlib.Path(__file__).parents[1] / "models" / "knn_recommendation_model.joblib"
DESTINATION  = f"models/recommendation_model/{MODEL_FILE.name}"

def main():
    client  = storage.Client(project=PROJECT_ID)
    bucket  = client.bucket(BUCKET_NAME)
    blob    = bucket.blob(DESTINATION)
    blob.upload_from_filename(MODEL_FILE)
    print(f"✅ Uploaded to gs://{BUCKET_NAME}/{DESTINATION}")

if __name__ == "__main__":
    main()
