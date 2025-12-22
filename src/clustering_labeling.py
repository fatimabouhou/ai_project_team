import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.stdout.reconfigure(encoding='utf-8')
# ============================================
# 1. CHARGER VOS DONNÉES
# ============================================
df = pd.read_csv('vehicle_features_cleaned.csv')

print("Dataset chargé:")
print(df.head())
print(f"\nNombre total de lignes: {len(df)}")

# ============================================
# 2. PRÉPARER LES FEATURES POUR LE CLUSTERING
# ============================================
# Sélectionner les features importantes
features = ['speed', 'relative_speed', 'acceleration', 
            'distance_to_nearest', 'direction_change']

X = df[features].copy()

# Gérer les valeurs manquantes si nécessaire
X = X.fillna(0)

print("\nStatistiques des features:")
print(X.describe())

# ============================================
# 3. NORMALISATION (TRÈS IMPORTANT!)
# ============================================
# Les features ont des échelles différentes (vitesse en km/h, distance en pixels)
# Il faut les normaliser pour que le clustering fonctionne bien

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nDonnées normalisées (moyenne=0, écart-type=1)")

# ============================================
# 4. TROUVER LE NOMBRE OPTIMAL DE CLUSTERS
# ============================================
# Méthode du coude (Elbow method)

inertias = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Visualiser
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Nombre de clusters (K)')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour trouver K optimal')
plt.grid(True)
plt.savefig('elbow_method.png')
plt.show()

print("\n💡 Regardez le graphique 'elbow_method.png'")
print("Choisissez K où la courbe fait un 'coude'")

# ============================================
# 5. APPLIQUER K-MEANS (K=3 pour Normal/Suspect/Dangereux)
# ============================================
K = 3  # Vous pouvez changer ce nombre selon le graphique ci-dessus

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n✅ Clustering terminé avec K={K}")
print("\nDistribution des clusters:")
print(df['cluster'].value_counts().sort_index())

# ============================================
# 6. ANALYSER CHAQUE CLUSTER
# ============================================
print("\n" + "="*60)
print("📊 ANALYSE DÉTAILLÉE DE CHAQUE CLUSTER")
print("="*60)

for i in range(K):
    cluster_data = df[df['cluster'] == i]
    print(f"\n{'='*60}")
    print(f"🔍 CLUSTER {i} - {len(cluster_data)} véhicules ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"{'='*60}")
    
    stats = cluster_data[features].describe()
    print("\nMoyennes:")
    print(f"  Vitesse moyenne: {cluster_data['speed'].mean():.2f} km/h")
    print(f"  Vitesse relative moyenne: {cluster_data['relative_speed'].mean():.2f} km/h")
    print(f"  Distance moyenne: {cluster_data['distance_to_nearest'].mean():.2f} px")
    print(f"  Accélération moyenne: {cluster_data['acceleration'].mean():.2f} km/h/s")
    print(f"  Changement direction moyen: {cluster_data['direction_change'].mean():.2f}°")
    
    print("\nMaximums:")
    print(f"  Vitesse max: {cluster_data['speed'].max():.2f} km/h")
    print(f"  Vitesse relative max: {cluster_data['relative_speed'].max():.2f} km/h")
    print(f"  Distance min: {cluster_data['distance_to_nearest'].min():.2f} px")

# ============================================
# 7. MAPPER LES CLUSTERS AUX LABELS
# ============================================
print("\n" + "="*60)
print("🎯 INTERPRÉTATION ET MAPPING DES CLUSTERS")
print("="*60)

# Calculer des scores de dangerosité pour chaque cluster
cluster_danger_scores = {}

for i in range(K):
    cluster_data = df[df['cluster'] == i]
    
    # Score basé sur plusieurs facteurs
    danger_score = 0
    
    # Plus la vitesse relative est élevée, plus c'est dangereux
    danger_score += cluster_data['relative_speed'].mean() * 2
    
    # Moins la distance est grande, plus c'est dangereux
    avg_distance = cluster_data['distance_to_nearest'].mean()
    if avg_distance > 0:
        danger_score += (1000 / avg_distance)  # Inverse de la distance
    
    # Plus l'accélération est forte, plus c'est dangereux
    danger_score += abs(cluster_data['acceleration'].mean()) * 3
    
    # Plus le changement de direction est brusque, plus c'est dangereux
    danger_score += abs(cluster_data['direction_change'].mean())
    
    cluster_danger_scores[i] = danger_score
    print(f"\nCluster {i} - Score de dangerosité: {danger_score:.2f}")

# Trier les clusters par score de dangerosité
sorted_clusters = sorted(cluster_danger_scores.items(), key=lambda x: x[1])

# Mapping automatique
cluster_to_label = {}
cluster_to_label[sorted_clusters[0][0]] = 0  # Le moins dangereux = Normal
cluster_to_label[sorted_clusters[1][0]] = 1  # Moyen = Suspect
cluster_to_label[sorted_clusters[2][0]] = 2  # Le plus dangereux = Dangereux

print("\n📋 MAPPING FINAL:")
for cluster_id, label in cluster_to_label.items():
    label_name = ['Normal', 'Suspect', 'Dangereux'][label]
    print(f"  Cluster {cluster_id} → Label {label} ({label_name})")

# Appliquer le mapping
df['danger_label'] = df['cluster'].map(cluster_to_label)

# ============================================
# 8. VISUALISATIONS
# ============================================
print("\n📊 Création des visualisations...")

# Visualisation 1: Distribution des labels
plt.figure(figsize=(8, 6))
label_counts = df['danger_label'].value_counts().sort_index()
colors = ['green', 'orange', 'red']
plt.bar(['Normal', 'Suspect', 'Dangereux'], label_counts.values, color=colors, alpha=0.7)
plt.xlabel('Catégorie de danger')
plt.ylabel('Nombre de véhicules')
plt.title('Distribution des labels de dangerosité')
plt.grid(axis='y', alpha=0.3)
plt.savefig('danger_distribution.png')
plt.show()

# Visualisation 2: Scatter plot vitesse vs distance
plt.figure(figsize=(10, 6))
for label, color, name in zip([0, 1, 2], ['green', 'orange', 'red'], 
                               ['Normal', 'Suspect', 'Dangereux']):
    subset = df[df['danger_label'] == label]
    plt.scatter(subset['speed'], subset['distance_to_nearest'], 
                c=color, label=name, alpha=0.5, s=20)

plt.xlabel('Vitesse (km/h)')
plt.ylabel('Distance au plus proche (pixels)')
plt.title('Vitesse vs Distance - Clustering de dangerosité')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('speed_vs_distance_clustering.png')
plt.show()

# Visualisation 3: Heatmap des moyennes par label
plt.figure(figsize=(10, 6))
label_means = df.groupby('danger_label')[features].mean()
sns.heatmap(label_means.T, annot=True, fmt='.2f', cmap='RdYlGn_r', 
            xticklabels=['Normal', 'Suspect', 'Dangereux'])
plt.title('Moyennes des features par catégorie de danger')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('features_heatmap.png')
plt.show()

# ============================================
# 9. SAUVEGARDER LES RÉSULTATS
# ============================================
# Sauvegarder le dataset avec les labels
df.to_csv('vehicle_features_labeled.csv', index=False)
print("\n✅ Dataset labellisé sauvegardé: vehicle_features_labeled.csv")

# Statistiques finales
print("\n" + "="*60)
print("📈 STATISTIQUES FINALES")
print("="*60)
print(f"\nTotal de véhicules: {len(df)}")
print(f"\nDistribution des labels:")
for label, name in enumerate(['Normal', 'Suspect', 'Dangereux']):
    count = (df['danger_label'] == label).sum()
    pct = count / len(df) * 100
    print(f"  {name}: {count} ({pct:.1f}%)")

# ============================================
# 10. EXEMPLES DE CHAQUE CATÉGORIE
# ============================================
print("\n" + "="*60)
print("📋 EXEMPLES DE CHAQUE CATÉGORIE")
print("="*60)

for label, name in enumerate(['Normal', 'Suspect', 'Dangereux']):
    print(f"\n{'='*60}")
    print(f"Exemples de comportement {name.upper()}:")
    print(f"{'='*60}")
    
    examples = df[df['danger_label'] == label].head(3)
    for idx, row in examples.iterrows():
        print(f"\n  Frame {row['frame']}, Véhicule #{row['tracker_id']}:")
        print(f"    Vitesse: {row['speed']:.1f} km/h")
        print(f"    Vitesse relative: {row['relative_speed']:.1f} km/h")
        print(f"    Distance: {row['distance_to_nearest']:.1f} px")
        print(f"    Accélération: {row['acceleration']:.1f} km/h/s")

print("\n" + "="*60)
print("✅ CLUSTERING TERMINÉ!")
print("="*60)
print("\nFichiers générés:")
print("  - vehicle_features_labeled.csv (dataset avec labels)")
print("  - elbow_method.png (choix du K optimal)")
print("  - danger_distribution.png (distribution des labels)")
print("  - speed_vs_distance_clustering.png (visualisation 2D)")
print("  - features_heatmap.png (moyennes par catégorie)")