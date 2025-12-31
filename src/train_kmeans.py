# =====================================================
# kmeans_simple.py - Clustering K-Means simple
# =====================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# -----------------------
# 1. Charger les données
# -----------------------
df = pd.read_csv("vehicle_features_labeled.csv")
features = ['speed', 'relative_speed', 'acceleration', 
            'distance_to_nearest', 'direction_change']
X = df[features].fillna(0)

# -----------------------
# 2. Normalisation
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------
# 3. Clustering K-Means
# -----------------------
k = 3  # nombre de clusters
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# -----------------------
# 4. Évaluation
# -----------------------
sil_score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {sil_score:.4f}")

# Distribution des clusters
unique, counts = np.unique(labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"Cluster {cluster_id}: {count} véhicules ({count/len(labels)*100:.1f}%)")

# Ajouter les labels au dataframe
df['kmeans_cluster'] = labels
df.to_csv("vehicle_features_with_clusters.csv", index=False)
print("✅ Clusters ajoutés au CSV : vehicle_features_with_clusters.csv")

# -----------------------
# 5. Visualisation 2D (PCA)
# -----------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:,0], 
            pca.transform(kmeans.cluster_centers_)[:,1],
            c='red', marker='X', s=200, label='Centres')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters K-Means (PCA 2D)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
