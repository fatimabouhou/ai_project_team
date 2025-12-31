import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
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
# 2. Split train/test par tracker
# -----------------------
trackers = df['tracker_id'].unique()
train_ids, test_ids = train_test_split(trackers, test_size=0.2, random_state=42)

train_mask = df['tracker_id'].isin(train_ids)
test_mask = df['tracker_id'].isin(test_ids)

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# -----------------------
# 3. Créer et entraîner le modèle
# -----------------------
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# -----------------------
# 4. Prédictions et évaluation
# -----------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------
# 5. Sauvegarde du modèle
# -----------------------
Path('outputs').mkdir(exist_ok=True)
joblib.dump(model, 'outputs/decision_tree_model.pkl')
print("✅ Modèle sauvegardé : outputs/decision_tree_model.pkl")

# -----------------------
# 6. Importance des features
# -----------------------
importances = model.feature_importances_
plt.figure(figsize=(8,5))
plt.bar(features, importances, color='skyblue')
plt.title("Importance des Features - Arbre de Décision")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/decision_tree_feature_importance.png")
plt.show()
print("✅ Graphique importance des features sauvegardé : outputs/decision_tree_feature_importance.png")

# -----------------------
# 7. (Optionnel) Visualiser l'arbre
# -----------------------
# export_graphviz(model, out_file="outputs/tree.dot", feature_names=features, class_names=['Normal','Suspect','Dangereux'], filled=True)
# Vous pouvez ensuite convertir le fichier .dot en image avec Graphviz :
# !dot -Tpng outputs/tree.dot -o outputs/tree.png
