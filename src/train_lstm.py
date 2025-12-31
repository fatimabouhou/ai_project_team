import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# -------------------------------
# 1. Chargement des données
# -------------------------------
df = pd.read_csv("vehicle_features_labeled.csv")

features = [
    'speed',
    'relative_speed',
    'avg_traffic_speed',
    'acceleration',
    'distance_to_nearest',
    'direction_change'
]

# -------------------------------
# 2. Normalisation des features
# -------------------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# -------------------------------
# 3. Fonction pour créer des séquences par tracker
# -------------------------------
def create_sequences_by_tracker(df, features, label_col, seq_len):
    X, y = [], []
    for tracker_id in df['tracker_id'].unique():
        group = df[df['tracker_id'] == tracker_id].sort_values('frame')
        X_values = group[features].values
        y_values = group[label_col].values
        if len(X_values) < seq_len:
            continue
        for i in range(len(X_values) - seq_len):
            X.append(X_values[i:i+seq_len])
            y.append(y_values[i+seq_len])
    return np.array(X), np.array(y)

# -------------------------------
# 4. Création des séquences
# -------------------------------
SEQ_LEN = 10
X, y = create_sequences_by_tracker(df, features, 'danger_label', SEQ_LEN)

# -------------------------------
# 5. Split par tracker
# -------------------------------
trackers = df['tracker_id'].unique()
train_ids, test_ids = train_test_split(trackers, test_size=0.2, random_state=42)

X_train, y_train, X_test, y_test = [], [], [], []

for tracker_id in train_ids:
    group = df[df['tracker_id'] == tracker_id].sort_values('frame')
    X_seq, y_seq = create_sequences_by_tracker(group, features, 'danger_label', SEQ_LEN)
    if len(X_seq) > 0:
        X_train.append(X_seq)
        y_train.append(y_seq)

for tracker_id in test_ids:
    group = df[df['tracker_id'] == tracker_id].sort_values('frame')
    X_seq, y_seq = create_sequences_by_tracker(group, features, 'danger_label', SEQ_LEN)
    if len(X_seq) > 0:
        X_test.append(X_seq)
        y_test.append(y_seq)

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

# -------------------------------
# 6. Construction du modèle LSTM
# -------------------------------
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, len(features))),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 7. Entraînement
# -------------------------------
history = model.fit(
    X_train,
    to_categorical(y_train),
    validation_data=(X_test, to_categorical(y_test)),
    epochs=40,
    batch_size=32
)

# -------------------------------
# 8. Évaluation
# -------------------------------
loss, acc = model.evaluate(X_test, to_categorical(y_test))
print(f"Test Accuracy: {acc:.4f}")

# -------------------------------
# 9. Sauvegarde du modèle et du scaler
# -------------------------------
Path('outputs').mkdir(exist_ok=True)
model.save('outputs/lstm_model.h5')
joblib.dump(scaler, 'outputs/lstm_scaler.pkl')
print("✅ Modèle LSTM sauvegardé : outputs/lstm_model.h5")
print("✅ Scaler sauvegardé : outputs/lstm_scaler.pkl")

# -------------------------------
# 10. Graphiques de loss et accuracy
# -------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/lstm_training_plots.png')
plt.show()
print("✅ Graphiques sauvegardés : outputs/lstm_training_plots.png")
