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
# Defining network boundaries
network_boundaries = {
    'VIS': (0, 24),
    'SMN': (24, 58),
    'DA': (58, 80),
    'SVAN': (80, 106),
    'FP': (106, 143),
    'DEF': (143, 186)
}

# Extract within and between network connections
def extract_within_between_connections(matrix, start, end):
    # Extract within network connections (submatrix)
    test = matrix[:, start:end, start:end]
    print('within_connec', test.shape)

    # Size calculations
    size_within = (end - start) ** 2
    size_upper = (start) * (end - start)
    size_lower = (186 - end) * (end - start)
    size_left = (end - start) * (start)
    size_right = (end - start) * (186 - end)

    # Dynamic batch size
    batch_size = tf.shape(test)[0]
    
    # Reshaping within_connections (flattening the submatrix)
    within_connections = tf.reshape(test, [batch_size, size_within])
    print('within_connections', within_connections.shape)

    # Extracting between network connections (cross shape)
    upper_rows = matrix[:, :start, start:end]
    print('upper_rows', upper_rows.shape)
    
    lower_rows = matrix[:, end:, start:end]
    print('lower_rows', lower_rows.shape)

    left_cols = matrix[:, start:end, :start]
    print('left_cols', left_cols.shape)

    right_cols = matrix[:, start:end, end:]
    print('right_cols', right_cols.shape)

    # Reshape between connections
    upper_rows_flat = tf.reshape(upper_rows, [batch_size, size_upper])
    lower_rows_flat = tf.reshape(lower_rows, [batch_size, size_lower])
    left_cols_flat = tf.reshape(left_cols, [batch_size, size_left])
    right_cols_flat = tf.reshape(right_cols, [batch_size, size_right])

    # Concatenating between connections
    between_connections = tf.concat([upper_rows_flat, lower_rows_flat, left_cols_flat, right_cols_flat], axis=1)
    print('between_connections', between_connections.shape)

    return within_connections, between_connections

# Define the subnetwork (Dense layers for each within and between connections)
def subnetwork(input_size, name):
    print('input_size',input_size)
    input_layer = layers.Input(shape=(input_size,), name=name + '_input')
    x = layers.Dense(64, activation='relu',kernel_initializer='he_normal')(input_layer)
    x = layers.Dense(32, activation='relu',kernel_initializer='he_normal')(x)
    x = layers.Dense(16, activation='relu',kernel_initializer='he_normal')(x)
    x = layers.Dense(8, activation='relu',kernel_initializer='he_normal')(x)
    output_layer = layers.Dense(6, activation='relu',kernel_initializer='he_normal')(x)  # Output 6 values
    return models.Model(inputs=input_layer, outputs=output_layer)

# Main Model
def create_model(input_shape):
    input_layer = layers.Input(shape=(input_shape,))
    #print('input_shape',input_shape)
    reshaped_input = layers.Reshape((186, 186))(input_layer)
    print('reshape_input',reshaped_input.shape)
    # List of subnetworks for each region
    within_outputs = []
    between_outputs = []

    for network, (start, end) in network_boundaries.items():
        within_conn, between_conn = extract_within_between_connections(reshaped_input, start, end)
        
        # Within-network submodel
        print('within_conn.shape',within_conn.shape)
        within_network_model = subnetwork(within_conn.shape[-1], network + '_within')
        within_output = within_network_model(within_conn)
        within_outputs.append(within_output)
        
        # Between-network submodel
        #print('between_conn.shape
        between_network_model = subnetwork(between_conn.shape[-1], network + '_between')
        between_output = between_network_model(between_conn)
        between_outputs.append(between_output)

    # Concatenate all within and between outputs
    concat_within = layers.Concatenate()(within_outputs)
    concat_between = layers.Concatenate()(between_outputs)
    combined = layers.Concatenate()([concat_within, concat_between])

    # Flatten and final output layer
    flat = layers.Flatten()(combined)
    output = layers.Dense(1, activation='linear', name='brain_age_output')(flat)

    model = models.Model(inputs=input_layer, outputs=output)
    return model

# Compile and train the model
def compile_and_train_model(X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):

    model = create_model(X_train.shape[1])
    print(model.summary())
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
    
    if X_val == None:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    else:
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return model, history

# Evaluation function
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    pcc = np.corrcoef(y_test, predictions.flatten())[0, 1]
    r2 = r2_score(y_test, predictions)

    return {'MAE': mae, 'RMSE': rmse, 'PCC': pcc, 'R2': r2}




def load_data(fold_id):
    X = np.load('X_fmri.npy')
    y = np.load('y_fmri.npy')
    train_indices = np.load(f'train_indices_fold{fold_id}.npy')
    val_indices   = np.load(f'val_indices_fold{fold_id}.npy')
    test_indices  = np.load(f'test_indices_fold{fold_id}.npy')
    
    return X[train_indices], X[val_indices], X[test_indices], y[train_indices], y[val_indices], y[test_indices]
    
def main():
    fold_id = sys.argv[1]
    #data loading
    #X_train, X_val, X_test, y_train, y_val, y_test = load_data(fold_id)
    X = np.load('X_fmri.npy') #needs to have dimensions of (N,186*186) --- where N is the number of patients and (186*186) is the np.flatten(matrix)
    y = np.load('y_fmri.npy') #needs to have dimensions of (N,)
    # Compile and train the model
    model, history = compile_and_train_model(X, y)
    model.save('model_new.h5')
    # Evaluate the model on the test data
    results = evaluate_model(model, X, y)
    print("Test Results:", results)

# # Execute main function
if __name__ == "__main__":
    main()
