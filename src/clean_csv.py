import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# =====================================================
# 1. FICHIERS
# =====================================================
input_file = 'vehicle_features.csv'
output_file_cleaned = 'vehicle_features_cleaned.csv'
output_file_summary = 'vehicle_features_summary.csv'

print("Chargement du fichier CSV...")
df = pd.read_csv(input_file)

print(f"✅ Données originales : {len(df)} lignes")
print(f"   Colonnes : {list(df.columns)}")
print(f"   Véhicules uniques : {df['tracker_id'].nunique()}")

# =====================================================
# 2. NETTOYAGE DES DONNÉES
# =====================================================
print("\n🧹 Nettoyage en cours...")

df_cleaned = df.copy()

# ---- Filtrer vitesses réalistes ----
df_cleaned = df_cleaned[df_cleaned['speed'].between(5, 200)]

# ---- Filtrer accélérations réalistes ----
df_cleaned = df_cleaned[df_cleaned['acceleration'].between(-20, 20)]

# ---- Filtrer distances réalistes ----
df_cleaned = df_cleaned[df_cleaned['distance_to_nearest'].between(1, 500)]

# ---- Supprimer premières frames de chaque véhicule ----
df_cleaned = df_cleaned.groupby('tracker_id').apply(
    lambda x: x.iloc[3:] if len(x) > 3 else x
).reset_index(drop=True)

# ---- Garder uniquement véhicules avec au moins 10 frames ----
vehicle_counts = df_cleaned['tracker_id'].value_counts()
valid_vehicles = vehicle_counts[vehicle_counts >= 10].index
df_cleaned = df_cleaned[df_cleaned['tracker_id'].isin(valid_vehicles)]

print(f"✅ Données nettoyées : {len(df_cleaned)} lignes")
print(f"   Véhicules restants : {df_cleaned['tracker_id'].nunique()}")

# ---- Sauvegarder CSV nettoyé ----
df_cleaned.to_csv(output_file_cleaned, index=False)
print(f"💾 Fichier nettoyé sauvegardé : {output_file_cleaned}")

# =====================================================
# 3. CRÉER UN RÉSUMÉ PAR VÉHICULE
# =====================================================
print("\n📊 Création du résumé par véhicule...")

summary = df_cleaned.groupby('tracker_id').agg({
    'vehicle_class': 'first',
    'speed': ['mean', 'max', 'min', 'std'],
    'acceleration': ['mean', 'max', 'min', 'std'],
    'distance_to_nearest': ['mean', 'min'],
    'direction_change': ['mean', 'max'],
    'relative_speed': ['mean', 'max', 'min'],
    'frame': 'count'
}).reset_index()

# Aplatir colonnes
summary.columns = [
    'tracker_id',
    'vehicle_class',
    'speed_mean',
    'speed_max',
    'speed_min',
    'speed_std',
    'acceleration_mean',
    'acceleration_max',
    'acceleration_min',
    'acceleration_std',
    'distance_mean',
    'distance_min',
    'direction_change_mean',
    'direction_change_max',
    'relative_speed_mean',
    'relative_speed_max',
    'relative_speed_min',
    'frames_count'
]

# Arrondir valeurs
for col in summary.columns:
    if col not in ['tracker_id', 'vehicle_class', 'frames_count']:
        summary[col] = summary[col].round(2)

# Sauvegarder résumé
summary.to_csv(output_file_summary, index=False)
print(f"💾 Résumé sauvegardé : {output_file_summary}")

# =====================================================
# 4. STATISTIQUES
# =====================================================
print("\n📈 STATISTIQUES APRÈS NETTOYAGE")
print(f"   Lignes totales : {len(df_cleaned)}")
print(f"   Véhicules uniques : {df_cleaned['tracker_id'].nunique()}")
print(f"   Vitesse moyenne : {df_cleaned['speed'].mean():.2f} km/h")
print(f"   Vitesse max : {df_cleaned['speed'].max():.2f} km/h")
print(f"   Accélération moyenne : {df_cleaned['acceleration'].mean():.2f} km/h/s")
print(f"   Distance moyenne : {df_cleaned['distance_to_nearest'].mean():.2f} px")
print(f"   Frames moyen par véhicule : {summary['frames_count'].mean():.1f}")
