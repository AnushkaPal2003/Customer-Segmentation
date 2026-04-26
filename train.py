import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import mlflow
import mlflow.sklearn

# =========================
# MLflow Setup (IMPORTANT)
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Customer_Segmentation")

# =========================
# Load Data
# =========================
print("Loading Data...")
df = pd.read_csv("Mall_Customers.csv")

features = ["Annual Income (k$)", "Spending Score (1-100)"]
x = df[features]

# =========================
# Scaling
# =========================
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# =========================
# Best K selection
# =========================
scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(x_scaled)
    scores.append(silhouette_score(x_scaled, labels))

best_k = np.argmax(scores) + 2
print("Best K:", best_k)

# =========================
# MLflow Tracking
# =========================
with mlflow.start_run(run_name="KMeans_Clustering"):

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = model.fit_predict(x_scaled)

    inertia = model.inertia_
    sil = silhouette_score(x_scaled, clusters)

    # Log params
    mlflow.log_param("model", "KMeans")
    mlflow.log_param("best_k", best_k)

    # Log metrics
    mlflow.log_metric("inertia", inertia)
    mlflow.log_metric("silhouette_score", sil)

    # Log model
    mlflow.sklearn.log_model(model, "model")

print("\nTraining Complete!")
print("Inertia:", inertia)
print("Silhouette:", sil)