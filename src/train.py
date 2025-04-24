"""
Training a simple K-Nearest-Neighbors recommender 
"""

import pathlib
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

DATA_PATH   = pathlib.Path(__file__).parents[1] / "data" / "sample_movies.csv"
MODEL_DIR   = pathlib.Path(__file__).parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE  = MODEL_DIR / "knn_recommendation_model.joblib"

def load_data(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)

def train_knn(df: pd.DataFrame):
    features = df[["feature_1", "feature_2"]]
    knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    knn.fit(features)
    return knn

def main():
    df  = load_data(DATA_PATH)
    mdl = train_knn(df)
    joblib.dump(mdl, MODEL_FILE)
    print(f"✅ Model saved to {MODEL_FILE.resolve()}")

if __name__ == "__main__":
    main()
