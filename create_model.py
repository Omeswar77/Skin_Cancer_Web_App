#create_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

# Dummy model (for testing UI)
model = Sequential([
    Flatten(input_shape=(224, 224, 3)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# Save model
model.save("model.h5")

print(" model.h5 created successfully!")