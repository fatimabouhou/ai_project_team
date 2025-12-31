import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Charger les données
df = pd.read_csv("vehicle_features_labeled.csv")

# Features et label
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

# Split par tracker pour éviter fuite de données
trackers = df['tracker_id'].unique()
train_ids, test_ids = train_test_split(trackers, test_size=0.2, random_state=42)

train_mask = df['tracker_id'].isin(train_ids)
test_mask = df['tracker_id'].isin(test_ids)

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# Créer le modèle XGBoost
model = XGBClassifier(
    n_estimators=200,      # nombre d’arbres
    max_depth=5,           # profondeur maximale des arbres
    learning_rate=0.1,     # taux d’apprentissage
    objective='multi:softmax',  # classification multi-classe
    num_class=3,           # nombre de classes
    random_state=42
)

# Entraîner
model.fit(X_train, y_train)

# Prédire
y_pred = model.predict(X_test)

# Évaluer
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib
from matplotlib import pyplot as plt
from xgboost import plot_importance

# Sauvegarde du modèle
joblib.dump(model, 'outputs/xgboost_model.pkl')
print("✅ Modèle XGBoost sauvegardé : outputs/xgboost_model.pkl")

# Importance des features
plt.figure(figsize=(10,6))
plot_importance(model)
plt.title("Importance des Features - XGBoost")
plt.tight_layout()
plt.savefig('outputs/xgboost_feature_importance.png')
plt.show()
print("✅ Graphique sauvegardé : outputs/xgboost_feature_importance.png")
