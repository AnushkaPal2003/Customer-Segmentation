# 🧠 Customer Segmentation using KMeans + MLflow + Docker

## 📌 Project Overview
This project performs customer segmentation using KMeans clustering on mall customer data. The goal is to group customers based on their Annual Income and Spending Score to help businesses target customers effectively.

The project also includes **MLflow for experiment tracking** and **Docker for containerization**, making it reproducible and deployment-ready.


## 📊 Problem Statement
Businesses want to understand customer behavior and segment them into meaningful groups to improve marketing strategies and customer engagement.


## ⚙️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- MLflow (Experiment Tracking)
- Docker (Containerization)


## 🧹 Data Preprocessing
- Checked and removed duplicates
- Verified missing values
- Outlier detection using IQR method
- Feature scaling using StandardScaler


## 🤖 Model Used
- KMeans Clustering
- Optimal clusters selected using Elbow Method
- Evaluation using:
  - Inertia
  - Silhouette Score


## 📈 MLflow Tracking
MLflow was used to:
- Log parameters (number of clusters)
- Track metrics (inertia, silhouette score)
- Store trained model as artifact
- Compare different runs


## 🐳 Docker Setup
The project is containerized using Docker to ensure consistent execution across environments.
