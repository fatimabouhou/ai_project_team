import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
# ============================================
# 1. CHARGER LE DATASET LABELLISÉ
# ============================================
df = pd.read_csv('vehicle_features_labeled.csv')

print("Dataset chargé:")
print(df.head(20))
print(f"\nNombre total de lignes: {len(df)}")
print(f"\nColonnes disponibles: {df.columns.tolist()}")

# ============================================
# 2. AFFICHAGE FORMATÉ DU DATASET
# ============================================
print("\n" + "="*120)
print("📊 VISUALISATION DU DATASET AVEC LABELS")
print("="*120)

# Créer un affichage formaté
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
pd.set_option('display.float_format', '{:.2f}'.format)

# Sélectionner les colonnes importantes
columns_to_show = ['frame', 'tracker_id', 'vehicle_class', 'speed', 
                   'acceleration', 'distance_to_nearest', 'direction_change', 
                   'relative_speed', 'avg_traffic_speed', 'danger_label']

# Renommer pour un affichage plus clair
df_display = df[columns_to_show].copy()
df_display['label_name'] = df_display['danger_label'].map({
    0: '✅ Normal',
    1: '⚠️  Suspect',
    2: '🚨 Dangereux'
})

print("\n📋 APERÇU DU DATASET (50 premières lignes):")
print(df_display.head(50).to_string(index=False))

# ============================================
# 3. STATISTIQUES PAR LABEL
# ============================================
print("\n" + "="*120)
print("📈 STATISTIQUES PAR CATÉGORIE DE DANGER")
print("="*120)

for label in [0, 1, 2]:
    label_name = ['✅ NORMAL', '⚠️  SUSPECT', '🚨 DANGEREUX'][label]
    subset = df[df['danger_label'] == label]
    
    print(f"\n{label_name} ({len(subset)} véhicules, {len(subset)/len(df)*100:.1f}%)")
    print("-" * 80)
    print(f"  Vitesse       : Min={subset['speed'].min():.1f}, Moy={subset['speed'].mean():.1f}, Max={subset['speed'].max():.1f} km/h")
    print(f"  Accélération  : Min={subset['acceleration'].min():.1f}, Moy={subset['acceleration'].mean():.1f}, Max={subset['acceleration'].max():.1f} km/h/s")
    print(f"  Distance      : Min={subset['distance_to_nearest'].min():.1f}, Moy={subset['distance_to_nearest'].mean():.1f}, Max={subset['distance_to_nearest'].max():.1f} px")
    print(f"  Zigzag        : Min={subset['direction_change'].min():.1f}, Moy={subset['direction_change'].mean():.1f}, Max={subset['direction_change'].max():.1f}°")
    print(f"  Vitesse rel.  : Min={subset['relative_speed'].min():.1f}, Moy={subset['relative_speed'].mean():.1f}, Max={subset['relative_speed'].max():.1f} km/h")

# ============================================
# 4. VISUALISATIONS GRAPHIQUES
# ============================================
print("\n📊 Création des visualisations détaillées...")

# Configuration des couleurs
colors = {0: 'green', 1: 'orange', 2: 'red'}
labels_names = {0: 'Normal', 1: 'Suspect', 2: 'Dangereux'}

# Figure 1: Grille de visualisations
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('📊 Analyse Complète des Features par Label', fontsize=16, fontweight='bold')

# 1. Distribution des vitesses
for label in [0, 1, 2]:
    subset = df[df['danger_label'] == label]
    axes[0, 0].hist(subset['speed'], bins=30, alpha=0.6, 
                    label=labels_names[label], color=colors[label])
axes[0, 0].set_xlabel('Vitesse (km/h)')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].set_title('Distribution des Vitesses')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Distribution des accélérations
for label in [0, 1, 2]:
    subset = df[df['danger_label'] == label]
    axes[0, 1].hist(subset['acceleration'], bins=30, alpha=0.6, 
                    label=labels_names[label], color=colors[label])
axes[0, 1].set_xlabel('Accélération (km/h/s)')
axes[0, 1].set_ylabel('Fréquence')
axes[0, 1].set_title('Distribution des Accélérations')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Distribution des distances
for label in [0, 1, 2]:
    subset = df[df['danger_label'] == label]
    axes[1, 0].hist(subset['distance_to_nearest'], bins=30, alpha=0.6, 
                    label=labels_names[label], color=colors[label])
axes[1, 0].set_xlabel('Distance (pixels)')
axes[1, 0].set_ylabel('Fréquence')
axes[1, 0].set_title('Distribution des Distances')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Distribution des changements de direction (zigzag)
for label in [0, 1, 2]:
    subset = df[df['danger_label'] == label]
    axes[1, 1].hist(subset['direction_change'], bins=30, alpha=0.6, 
                    label=labels_names[label], color=colors[label])
axes[1, 1].set_xlabel('Changement de direction (°)')
axes[1, 1].set_ylabel('Fréquence')
axes[1, 1].set_title('Distribution des Zigzags')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 5. Distribution des vitesses relatives
for label in [0, 1, 2]:
    subset = df[df['danger_label'] == label]
    axes[2, 0].hist(subset['relative_speed'], bins=30, alpha=0.6, 
                    label=labels_names[label], color=colors[label])
axes[2, 0].set_xlabel('Vitesse Relative (km/h)')
axes[2, 0].set_ylabel('Fréquence')
axes[2, 0].set_title('Distribution des Vitesses Relatives')
axes[2, 0].legend()
axes[2, 0].grid(alpha=0.3)

# 6. Boxplot comparatif de toutes les features
features_normalized = df[['speed', 'acceleration', 'distance_to_nearest', 
                          'direction_change', 'relative_speed']].copy()
# Normaliser pour comparer sur le même graphique
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_normalized_values = scaler.fit_transform(features_normalized)
features_normalized = pd.DataFrame(features_normalized_values, 
                                   columns=features_normalized.columns)
features_normalized['danger_label'] = df['danger_label'].values

data_for_box = []
labels_for_box = []
colors_for_box = []
for feature in ['speed', 'acceleration', 'distance_to_nearest', 'direction_change', 'relative_speed']:
    for label in [0, 1, 2]:
        subset = features_normalized[features_normalized['danger_label'] == label]
        data_for_box.append(subset[feature].values)
        labels_for_box.append(f"{feature[:6]}\n{labels_names[label]}")
        colors_for_box.append(colors[label])

bp = axes[2, 1].boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
for patch, color in zip(bp['boxes'], colors_for_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[2, 1].set_ylabel('Valeur normalisée')
axes[2, 1].set_title('Comparaison de toutes les features (normalisées)')
axes[2, 1].tick_params(axis='x', rotation=45, labelsize=8)
axes[2, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('dataset_visualization_complete.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 5. SCATTER PLOTS DÉTAILLÉS
# ============================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('🎯 Relations entre Features et Labels', fontsize=16, fontweight='bold')

# 1. Vitesse vs Distance
for label in [0, 1, 2]:
    subset = df[df['danger_label'] == label]
    axes2[0, 0].scatter(subset['speed'], subset['distance_to_nearest'], 
                       c=colors[label], label=labels_names[label], alpha=0.4, s=10)
axes2[0, 0].set_xlabel('Vitesse (km/h)')
axes2[0, 0].set_ylabel('Distance (pixels)')
axes2[0, 0].set_title('Vitesse vs Distance')
axes2[0, 0].legend()
axes2[0, 0].grid(alpha=0.3)

# 2. Vitesse vs Accélération
for label in [0, 1, 2]:
    subset = df[df['danger_label'] == label]
    axes2[0, 1].scatter(subset['speed'], subset['acceleration'], 
                       c=colors[label], label=labels_names[label], alpha=0.4, s=10)
axes2[0, 1].set_xlabel('Vitesse (km/h)')
axes2[0, 1].set_ylabel('Accélération (km/h/s)')
axes2[0, 1].set_title('Vitesse vs Accélération')
axes2[0, 1].legend()
axes2[0, 1].grid(alpha=0.3)

# 3. Distance vs Accélération
for label in [0, 1, 2]:
    subset = df[df['danger_label'] == label]
    axes2[1, 0].scatter(subset['distance_to_nearest'], subset['acceleration'], 
                       c=colors[label], label=labels_names[label], alpha=0.4, s=10)
axes2[1, 0].set_xlabel('Distance (pixels)')
axes2[1, 0].set_ylabel('Accélération (km/h/s)')
axes2[1, 0].set_title('Distance vs Accélération')
axes2[1, 0].legend()
axes2[1, 0].grid(alpha=0.3)

# 4. Vitesse relative vs Zigzag
for label in [0, 1, 2]:
    subset = df[df['danger_label'] == label]
    axes2[1, 1].scatter(subset['relative_speed'], subset['direction_change'], 
                       c=colors[label], label=labels_names[label], alpha=0.4, s=10)
axes2[1, 1].set_xlabel('Vitesse Relative (km/h)')
axes2[1, 1].set_ylabel('Changement de Direction (°)')
axes2[1, 1].set_title('Vitesse Relative vs Zigzag')
axes2[1, 1].legend()
axes2[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scatter_plots_features.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 6. HEATMAP DE CORRÉLATION PAR LABEL
# ============================================
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle('🔥 Corrélation entre Features par Catégorie', fontsize=16, fontweight='bold')

features_for_corr = ['speed', 'acceleration', 'distance_to_nearest', 
                     'direction_change', 'relative_speed']

for idx, label in enumerate([0, 1, 2]):
    subset = df[df['danger_label'] == label][features_for_corr]
    corr_matrix = subset.corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=axes3[idx], cbar_kws={'label': 'Corrélation'})
    axes3[idx].set_title(f'{labels_names[label]}', fontweight='bold')

plt.tight_layout()
plt.savefig('correlation_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 7. TABLEAU RÉCAPITULATIF EXPORTÉ
# ============================================
print("\n" + "="*120)
print("💾 EXPORT DES RÉSULTATS")
print("="*120)

# Créer un tableau récapitulatif
summary = df.groupby('danger_label').agg({
    'speed': ['min', 'mean', 'max', 'std'],
    'acceleration': ['min', 'mean', 'max', 'std'],
    'distance_to_nearest': ['min', 'mean', 'max', 'std'],
    'direction_change': ['min', 'mean', 'max', 'std'],
    'relative_speed': ['min', 'mean', 'max', 'std'],
    'tracker_id': 'count'
}).round(2)

summary.to_csv('statistics_by_label.csv')
print("\n✅ Statistiques sauvegardées: statistics_by_label.csv")

# Sauvegarder un échantillon de chaque catégorie
for label in [0, 1, 2]:
    subset = df[df['danger_label'] == label].head(100)
    filename = f'sample_{labels_names[label].lower()}.csv'
    subset.to_csv(filename, index=False)
    print(f"✅ Échantillon {labels_names[label]}: {filename}")

print("\n" + "="*120)
print("✅ VISUALISATION TERMINÉE!")
print("="*120)
print("\nFichiers générés:")
print("  - dataset_visualization_complete.png (distributions de toutes les features)")
print("  - scatter_plots_features.png (relations entre features)")
print("  - correlation_heatmaps.png (corrélations par catégorie)")
print("  - statistics_by_label.csv (statistiques détaillées)")
print("  - sample_normal.csv, sample_suspect.csv, sample_dangereux.csv")