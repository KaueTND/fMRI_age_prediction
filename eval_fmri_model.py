# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:26:32 2024

@author: kaueu
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load your trained model
path = "."
model = tf.keras.models.load_model(path+'model.h5')  # Replace with your model path



# Define a function to extract the combined layer output
def get_combined_output(model):
    # Find the combined layer by its name (adjust if your name is different)
    combined_layer = model.get_layer(name="flatten")  # Replace with actual combined layer name
    # Create a model that outputs the combined layer
    combined_model = tf.keras.models.Model(inputs=model.input, outputs=combined_layer.output)
    return combined_model

# Main function to compute neuron importance and prediction errors
def get_neuron_importance_and_prediction_errors(model, input_matrices, y_test):
    num_samples = input_matrices.shape[0]
    num_neurons = 72  # Assuming combined layer has 72 neurons

    # Prepare arrays to store results
    importance_scores_batch = np.zeros((num_samples, num_neurons))
    highlighted_neurons_batch = np.zeros((num_samples, num_neurons), dtype=int)
    predictions = np.zeros(num_samples)
    prediction_errors = np.zeros(num_samples)

    # Extract the combined output model
    combined_model = get_combined_output(model)

    # Loop through each matrix
    for i in range(num_samples):
        input_matrix = input_matrices[i].reshape((1, -1))  # Reshape individual matrix for model input

        # Predict output and combined layer output
        predictions[i] = model.predict(input_matrix, verbose=0)[0][0]
        combined_output_value = combined_model.predict(input_matrix, verbose=0)
        # Convert combined output to a tensor for gradient tracking
        combined_output_tensor = tf.convert_to_tensor(combined_output_value, dtype=tf.float32)

        # Calculate prediction error
        prediction_errors[i] = predictions[i] - y_test[i]

        # Define a model to get gradients of the prediction w.r.t. combined layer output
        with tf.GradientTape() as tape:
            tape.watch(combined_output_tensor)
            prediction = model(input_matrix, training=False)
        
        # Compute gradients of the prediction w.r.t combined layer neurons
        grads = tape.gradient(prediction, combined_output_tensor)

        if grads is not None:  # Ensure gradients were computed
            # Absolute value of gradients for neuron importance
            importance_scores = tf.abs(grads[0]).numpy()
            
            # Normalize scores
            importance_scores /= importance_scores.sum()

            # Sort indices by importance in descending order
            highlighted_neurons = np.argsort(-importance_scores)
            
            # Store results
            importance_scores_batch[i] = importance_scores
            highlighted_neurons_batch[i] = highlighted_neurons
        else:
            print(f"Gradient computation failed for sample {i}.")

    return importance_scores_batch, highlighted_neurons_batch, predictions, prediction_errors

# Usage with a batch of matrices and test labels (replace with actual data)
X = np.load('X_fmri.npy') #needs to have dimensions of (N,186*186) --- where N is the number of patients and (186*186) is the np.flatten(matrix)
y = np.load('y_fmri.npy') #needs to have dimensions of (N,)
importance_scores_batch, highlighted_neurons_batch, predictions, prediction_errors = get_neuron_importance_and_prediction_errors(model, X, y)

# Display results
print("Importance Scores for Each Neuron in Each Matrix:", importance_scores_batch)
print("Most Important Neurons (indices) for Each Matrix:", highlighted_neurons_batch)
print("Predictions:", predictions)
print("Prediction Errors (Predicted - True):", prediction_errors)

# Optionally compute overall metrics
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
print(f"MAE: {mae}, RMSE: {rmse}")

# Save the arrays
np.save(os.path.join(path, "importance_scores_batch.npy"), importance_scores_batch)
np.save(os.path.join(path, "highlighted_neurons_batch.npy"), highlighted_neurons_batch)
np.save(os.path.join(path, "predictions.npy"), predictions)
np.save(os.path.join(path, "prediction_errors.npy"), prediction_errors)

# Optionally save the evaluation metrics
np.save(os.path.join(path, "MAE.npy"), np.array([mae]))
np.save(os.path.join(path, "RMSE.npy"), np.array([rmse]))

print("Results saved successfully in:", path)
