import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# 1. Chargement des données
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

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# -----------------------
# 4. Création du modèle MLP
# -----------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------
# 5. Entraînement
# -----------------------
history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=40,
    batch_size=32
)

# -----------------------
# 6. Évaluation
# -----------------------
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------
# 7. Création dossier pour sauvegarde
# -----------------------
Path('outputs').mkdir(exist_ok=True)

# -----------------------
# 8. Sauvegarde du modèle
# -----------------------
model.save("outputs/mlp_model.h5")
print("✅ Modèle sauvegardé : outputs/mlp_model.h5")

# -----------------------
# 9. Sauvegarde du scaler
# -----------------------
import joblib
joblib.dump(scaler, "outputs/scaler.save")
print("✅ Scaler sauvegardé : outputs/scaler.save")

# -----------------------
# 10. Graphiques de l'entraînement
# -----------------------
# Accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy pendant l\'entraînement')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("outputs/accuracy_plot.png")
plt.close()
print("✅ Graphique accuracy sauvegardé : outputs/accuracy_plot.png")

# Loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss pendant l\'entraînement')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("outputs/loss_plot.png")
plt.close()
print("✅ Graphique loss sauvegardé : outputs/loss_plot.png")
