# load data 

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


def load_data(csv_path, time_steps, train_split=0.8, feature_columns=None, target_column='Target'):
    """
    Load, preprocess, and split data into training and testing sets for LSTM model.

    Args:
        csv_path (str): Path to the CSV file.
        time_steps (int): Number of time steps to use for sequences.
        train_split (float): Fraction of the data to use for training (0.8 = 80% training, 20% testing).
        feature_columns (list): List of feature columns to use. Defaults to None, meaning all but the target.
        target_column (str): Column name of the target variable.

    Returns:
        X_train (torch.Tensor): Preprocessed feature tensors for training.
        y_train (torch.Tensor): Target tensors for training.
        X_test (torch.Tensor): Preprocessed feature tensors for testing.
        y_test (torch.Tensor): Target tensors for testing.
        target_scaler (MinMaxScaler): Fitted scaler for the target column (for inverse transforming predictions).
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # If no specific feature columns are provided, use all columns except target, Date, Ticker
    if feature_columns is None:
        feature_columns = df.columns.drop([target_column, 'Date', 'Ticker'])

    # Normalize the feature columns using StandardScaler
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(df[feature_columns])

    # Normalize the target column separately using StandardScaler
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(df[[target_column]])
        # Prepare sequences for LSTM (sliding window approach)
    X, y = [], []
    for i in range(time_steps, len(df)):
        X.append(scaled_features[i-time_steps:i])
        y.append(scaled_target[i])  # Next time-step target

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)  # Ensure y is 2D

    # Perform a time-based train-test split
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, target_scaler

# double check to see if the data loader worked: 
csv_path = 'Data/processed_stock_data.csv'
time_steps = 10
train_split = 0.8

# Load the data using the load_data function
train_features, train_targets, test_features, test_targets, target_scaler = load_data(csv_path, time_steps, train_split)

# lets see what the minmax scalar does

def plot_scaled_target(csv_path, target_column='Target'):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Display original target values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df[target_column])
    plt.title('Original Target Values')
    plt.ylabel('Value')
    plt.xlabel('Time Steps')

    # Apply MinMaxScaler to the target column
    scaler = StandardScaler()
    df[target_column + '_Scaled'] = scaler.fit_transform(df[[target_column]])

    # Display scaled target values
    plt.subplot(1, 2, 2)
    plt.plot(df[target_column + '_Scaled'])
    plt.title('Scaled Target Values')
    plt.ylabel('Scaled Value')
    plt.xlabel('Time Steps')

    plt.tight_layout()
    plt.show()

plot_scaled_target(csv_path)

