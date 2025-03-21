import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
df = pd.read_csv('Player Predictions/cleaned_dataset.csv')
df = df.sort_values(by='game_date')
df = df.drop(columns=['Unnamed: 0','team_score', 'opponent_team_score', 'athlete_id'])
df = df[df['target'] > 0]

# Reset index after filtering
df = df.reset_index(drop=True)

# Remove date column if it won't be used in the model
df = df.drop(columns=['game_date'])

# Handle remaining missing values if any
df = df.fillna(0)

# Split the data
# Use a temporal split since this is time series data
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Define features and target
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

# Define evaluation function
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

# 1. LightGBM Model
print("Training LightGBM Model...")
params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'verbose': -1
}

# Handle categorical features
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    for col in categorical_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

lgb_model = lgb.LGBMRegressor(**params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse'
)

# Make predictions
lgb_preds = lgb_model.predict(X_test)
lgb_metrics = evaluate_model(y_test, lgb_preds, "LightGBM")

# Feature importance
lgb_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Important Features:")
print(lgb_importance.head(20))

# 2. Neural Network Model
print("\nTraining Neural Network Model...")

# Standardize features for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
def build_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

nn_model = build_nn_model(X_train_scaled.shape[1])

# Train with early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = nn_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions with neural network
nn_preds = nn_model.predict(X_test_scaled).flatten()
nn_metrics = evaluate_model(y_test, nn_preds, "Neural Network")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Compare predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lgb_preds, alpha=0.5, label='LightGBM')
plt.scatter(y_test, nn_preds, alpha=0.5, label='Neural Network')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Actual vs Predicted Points')
plt.legend()
plt.grid(True)
plt.savefig('model_comparison.png')
plt.show()

# Compare model performance
models = ['LightGBM', 'Neural Network']
metrics = ['rmse', 'mae', 'r2']
results = pd.DataFrame({
    'LightGBM': [lgb_metrics['rmse'], lgb_metrics['mae'], lgb_metrics['r2']],
    'Neural Network': [nn_metrics['rmse'], nn_metrics['mae'], nn_metrics['r2']]
}, index=metrics)

print("\nModel Comparison:")
print(results)

# Optional: Save the models
lgb_model.booster_.save_model('lightgbm_player_points.txt')
nn_model.save('neural_network_player_points.h5')

print("\nModels trained and evaluated successfully!")