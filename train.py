import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

# =========================
# 1. Load Data
# =========================
print('\nLoading Data...')
df = pd.read_csv('Mall_Customers.csv')

print("Original Data:", df.shape)
print(df.isnull().sum())
print(df.duplicated().sum())

df = df.dropna().drop_duplicates()
print("Data after cleaning:", df.shape)

print("\nStatistics:")
print(df.describe())

# =========================
# 2. Visualization
# =========================
plt.figure(figsize=(6,4))
sns.boxplot(x='Annual Income (k$)', data=df)
plt.title('Annual Income')
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
plt.title('Income vs Spending Score')
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title('Correlation Heatmap')
plt.show()

# =========================
# 3. Feature Selection
# =========================
features = ['Annual Income (k$)', 'Spending Score (1-100)']
x = df[features]

# =========================
# 4. Outlier Removal (IQR)
# =========================
out_count = 0

for col in features:
    q1 = x[col].quantile(0.25)
    q3 = x[col].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr

    before = x.shape[0]
    x = x[(x[col] >= lb) & (x[col] <= ub)]
    after = x.shape[0]
    out_count += (before - after)

print("\nTotal outliers removed:", out_count)

# =========================
# 5. Scaling
# =========================
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# =========================
# 6. Elbow Method
# =========================
inert = []

for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(x_scaled)
    inert.append(km.inertia_)

plt.plot(range(1, 11), inert, marker='o')
plt.xlabel("Clusters")
plt.ylabel("Inertia")
plt.title("KMeans Elbow Method")
plt.show()

# =========================
# 7. MLflow Training
# =========================
mlflow.set_experiment("Customer_Segmentation")

with mlflow.start_run():
    model = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = model.fit_predict(x_scaled)

    inertia = model.inertia_

    mlflow.log_param("n_clusters", 5)
    mlflow.log_metric("inertia", inertia)
    mlflow.log_metric("outliers_removed", out_count)

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    mlflow.sklearn.log_model(model, "kmeans-model")

print("\nModel trained and saved.")
print("Final Inertia:", inertia)

# =========================
# 8. Cluster Visualization
# =========================
plt.figure(figsize=(6,5))
plt.scatter(
    x_scaled[:, 0],
    x_scaled[:, 1],
    c=clusters,
    cmap='viridis'
)
plt.xlabel("Scaled Income")
plt.ylabel("Scaled Spending Score")
plt.title("Customer Segments")
plt.show()
