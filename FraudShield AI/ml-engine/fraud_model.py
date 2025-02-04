# ml-engine/fraud_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic data as a placeholder for historical transaction features and labels
data = np.random.rand(1000, 10)  # 1000 transactions, 10 features each
labels = np.random.randint(2, size=(1000, 1))  # Binary labels: 0 (legit) or 1 (fraud)

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, labels, epochs=10, batch_size=32)

# Save the trained model for later inference in production
model.save("fraud_model.h5")
print("Model saved as fraud_model.h5")
