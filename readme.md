# K-Nearest-Neighbors Movie Recommender on Vertex AI (Private Endpoint)

> End-to-end example showing how to train, version, and serve a scikit-learn  
> model **inside a VPC-only environment** using Google Cloud Vertex AI.

## What’s inside
| Folder | Purpose |
| ------ | ------- |
| `data/` | Tiny sample dataset (titles + two numeric features) |
| `notebooks/` | Optional exploratory notebook (no secret creds) |
| `src/` | Python modules for training, uploading, deploying, and testing |
| `models/` | _Ignored_ by Git—artifacts live in Cloud Storage |

## Quick start

```bash
conda create -n knn python=3.11
conda activate knn
pip install -r requirements.txt

# 1️⃣ Train locally
python src/train.py

# 2️⃣ Upload to GCS
export GOOGLE_CLOUD_PROJECT=<your-project>
export MODEL_BUCKET=vertexai-sdh
python src/upload_to_gcs.py

# 3️⃣ Deploy to Vertex AI
export VPC_NETWORK_URI="projects/<proj>/global/networks/<vpc-name>"
python src/deploy_vertex.py
