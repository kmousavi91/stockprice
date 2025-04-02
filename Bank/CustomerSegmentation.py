import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate a synthetic dataset to simulate customer banking data
np.random.seed(42)
n_samples = 500
X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.5, random_state=42)

# Create a DataFrame with relevant features
df = pd.DataFrame(X, columns=['Income', 'Account Balance'])
df['Transaction Frequency'] = np.random.randint(1, 20, size=n_samples)
df['Loan Status'] = np.random.choice([0, 1], size=n_samples)  # 0: No Loan, 1: Has Loan
df['Age'] = np.random.randint(18, 70, size=n_samples)

# Normalize the data for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# Finding the optimal number of clusters using the Elbow Method
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# Applying K-Means with the optimal number of clusters (assumed k=4 from the Elbow Method)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Display cluster characteristics
cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)

# Visualize customer segmentation
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Income'], y=df['Account Balance'], hue=df['Cluster'], palette='viridis')
plt.title('Customer Segmentation Based on Income & Account Balance')
plt.xlabel('Income')
plt.ylabel('Account Balance')
plt.show()

