#Build a dense neural network to accurately detect the particle.
#The goal is to maximize your accuracy.
#Include a discussion of how you know your model has finished training 
#as well as what design decisions you made while building the network.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import gzip
import json

# check progress
print('imports done')

def peek_into_gz(file_path, max_lines=5):
    """
    Peek into the contents of a gzip-compressed file.
    Print the first few lines (or maximum lines specified) without fully decompressing the file.
    """
    with gzip.open(file_path, 'rt') as gz_file:
        for i, line in enumerate(gz_file):
            if i < max_lines:
                print(line.rstrip())  # Print each line, removing trailing newline
            else:
                print("...")
                break

# Path to your .gz file
gz_file_path = "all_train.csv.gz"

# Peek into the contents of the .gz file
# peek_into_gz(gz_file_path)

# Open the .gz file in read mode
with gzip.open(gz_file_path, 'rt') as gz_file:
    # Read the contents of the gzip file into a pandas DataFrame
    data0 = pd.read_csv(gz_file)

# Display the DataFrame
print(data0)
print(data0.columns)

## Separate features and target variable
X = data0.drop('# label', axis=1)  # Replace 'target_column' with your actual class label column
y = data0['# label']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_seq_model(optimizer=None):
    # Define your model
    model = Sequential([
        Dense(64, activation='relu'),  # Adjust input shape based on X_train
        Dropout(0.5),  # Add dropout for regularization
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Assuming binary classification
    ])

    if optimizer is None:
        optimizer = Adam(learning_rate=0.001)  # Default optimizer if not provided
  
    # # Compile the model
    # model.compile(optimizer=Adam(learning_rate=0.001),
    #             loss='binary_crossentropy',
    #             metrics=['accuracy'])
    return model

# Define EarlyStopping callback for models
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# create KerasClassifier for model creation
model = KerasClassifier(build_fn=create_seq_model, verbose=0)

# create parameter grid for GridSearchCV
param_grid = {
    'optimizer': ['adam', 'rmsprop']#,
    # 'dropout_rate': [0.3, 0.5, 0.7],
    # Add other hyperparameters to tune
}

# check progress
print('pre-gridsearch')

# perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train, callbacks=[early_stopping])

# check progress
print('gridsearch done')
# print('gridsearch skipped')

# Access best parameters
seq_best_params = grid_result.best_params_

# Create a dictionary for combined storage
best_params = {
    'Sequential': seq_best_params
    # 'SGD': sgd_best_params
}

# Save to a JSON file
with open('best_params.json', 'w') as f:
    json.dump(best_params, f)

with open('best_params.json', 'r') as f:
    loaded_params = json.load(f)

# check progress
print('best params saved')

# Access individual model's best parameters
seq_best_params = loaded_params['Sequential']
# best_sgd_params = loaded_params['SGD']

# Train the model
model = create_seq_model(**seq_best_params)  # Unpack best_params as function arguments
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
# history = create_seq_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# check progress
print('model trained')

# Evaluate the model on test data
test_loss, test_accuracy = create_seq_model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# Plotting training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Discussion:
# 1. Network Architecture:
#    - I used a simple architecture with fully connected (dense) layers.
#    - ReLU activation is used for hidden layers and sigmoid activation for the output layer for binary classification.
#    - Dropout layers are added to prevent overfitting.
# 2. Optimization:
#    - Adam optimizer is chosen for its adaptive learning rate properties.
#    - Binary cross-entropy loss is used as it is suitable for binary classification problems.
# 3. Training:
#    - The model is trained for 50 epochs with a batch size of 32.
#    - Validation split of 0.2 is used to monitor validation performance during training.
# 4. Evaluation:
#    - The model is evaluated on the test set to assess its generalization performance.
# 5. Monitoring Training Completion:
#    - Training completion can be judged by observing the convergence of training and validation accuracy.
#    - Additionally, monitoring the loss curves helps to ensure the model is not overfitting.

# Note: Replace preprocess_data() with your actual data loading and preprocessing function.
