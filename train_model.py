import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

# Load Dataset
df = pd.read_csv(
   r"C:\Users\Asmithaa\.cache\kagglehub\datasets\sachinpatel21\az-handwritten-alphabets-in-csv-format\versions\5\A_Z Handwritten Data.csv"
    )
print(df.head())

# Labels
y = df.iloc[:, 0].values

# Pixels
X = df.iloc[:, 1:].values

# Normalize
X = X / 255.0

# Reshape
X = X.reshape(-1, 28, 28, 1)

# One Hot Encoding
y = tf.keras.utils.to_categorical(y, 26)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# CNN Model
model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(28,28,1)
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        26,
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nAccuracy:", accuracy)

model.save(
    "models/character_model.h5"
)

print("Character Model Saved Successfully!")