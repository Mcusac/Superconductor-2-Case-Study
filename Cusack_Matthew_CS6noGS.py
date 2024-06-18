#Build a dense neural network to accurately detect the particle.
#The goal is to maximize your accuracy.
#Include a discussion of how you know your model has finished training 
#as well as what design decisions you made while building the network.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import gzip

# Load data
gz_file_path = "all_train.csv.gz"
with gzip.open(gz_file_path, 'rt') as gz_file:
    data0 = pd.read_csv(gz_file)

# Separate features and target variable
X = data0.drop('# label', axis=1)
y = data0['# label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
print("Precision:", precision)
print("Recall:", recall)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Get the weights of the first layer
first_layer_weights = model.layers[0].get_weights()[0]

# Calculate the absolute sum of weights for each feature
feature_importance = abs(first_layer_weights).sum(axis=1)

# Get the feature names
feature_names = X.columns

# Sort features based on importance
sorted_indices = feature_importance.argsort()[::-1]
sorted_feature_names = feature_names[sorted_indices]
sorted_feature_importance = feature_importance[sorted_indices]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names[:10], sorted_feature_importance[:10])
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.show()
