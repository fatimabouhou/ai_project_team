import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------
# 1. Charger les données
# -----------------------
df = pd.read_csv("vehicle_features_labeled.csv")

features = [
    'speed',
    'relative_speed',
    'avg_traffic_speed',
    'acceleration',
    'distance_to_nearest',
    'direction_change'
]

X = df[features].values
y = df['danger_label'].values

# -----------------------
# 2. Normalisation
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------
# 3. Split par tracker
# -----------------------
trackers = df['tracker_id'].unique()
train_ids, test_ids = train_test_split(trackers, test_size=0.2, random_state=42)

train_mask = df['tracker_id'].isin(train_ids)
test_mask = df['tracker_id'].isin(test_ids)

X_train = X_scaled[train_mask]
y_train = y[train_mask]
X_test = X_scaled[test_mask]
y_test = y[test_mask]

# -----------------------
# 4. Créer et entraîner le modèle KNN
# -----------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# -----------------------
# 5. Prédictions et évaluation
# -----------------------
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# -----------------------
# 6. Sauvegarde du modèle et du scaler
# -----------------------
Path('outputs').mkdir(exist_ok=True)
joblib.dump(knn, 'outputs/knn_model.pkl')
joblib.dump(scaler, 'outputs/knn_scaler.pkl')
print("✅ Modèle KNN sauvegardé : outputs/knn_model.pkl")
print("✅ Scaler sauvegardé : outputs/knn_scaler.pkl")

# -----------------------
# 7. Visualisation de la matrice de confusion
# -----------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN - Confusion Matrix')
plt.tight_layout()
plt.savefig('outputs/knn_confusion_matrix.png')
plt.show()
print("✅ Graphique matrice de confusion sauvegardé : outputs/knn_confusion_matrix.png")
