import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('Game Predictions/cleaned_dataset.csv')

# Sort data chronologically
df = df.sort_values(by='game_date')

# Split train and test sets based on date
split_date = '2025-02-01'
train_df = df[df['game_date'] <= split_date].copy()
test_df = df[df['game_date'] > split_date].copy()

# Define the columns to use as features (excluding non-numerical columns)
excluded_columns = ['Unnamed: 0', 'Unnamed: 0.1', 'game_id', 'game_date', 'home_team', 'home_color', 'away_team', 'away_color', 'target']
X_train = train_df.drop(columns=excluded_columns).fillna(0)
y_train = train_df['target'].astype(int)

X_test = test_df.drop(columns=excluded_columns).fillna(0)
y_test = test_df['target'].astype(int)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM model (5 previous games as sequence)
def create_sequences(X, y, sequence_length=5):
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])  # Select previous 5 games
        y_seq.append(y.iloc[i])  # Target for the current game
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("\nTraining LSTM model with past 5 games as sequences...")
model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_data=(X_test_seq, y_test_seq))

# Save the model
model.save('Game Predictions/models/lstm_model.h5')

# Evaluate the LSTM model
y_pred_proba = model.predict(X_test_seq)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate metrics
accuracy = accuracy_score(y_test_seq, y_pred)
precision = precision_score(y_test_seq, y_pred)
recall = recall_score(y_test_seq, y_pred)
f1 = f1_score(y_test_seq, y_pred)
auc = roc_auc_score(y_test_seq, y_pred_proba)

# Print evaluation metrics
print(f"\nLSTM Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {auc:.4f}")
