from data_loader import load_data
from model import create_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_


# Specify the CSV file path
csv_file_path = 'Data/processed_stock_data.csv'

# Configuration parameters
input_size = 13  # Number of features
hidden_size = 128
num_layers = 3  # Number of LSTM layers
dropout_rate = 0.5 # started with 0.2
output_size = 1  # We are predicting one output (e.g., closing price difference)
time_steps = 1  # Number of time window for the sequences
learning_rate = 0.001
epochs = 100  # Number of training epochs
batch_size = 25  # Define batch size for training

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training and testing data
train_features, train_targets, test_features, test_targets, target_scaler = load_data(csv_file_path, time_steps)
train_features, train_targets = train_features.to(device), train_targets.to(device)
test_features, test_targets = test_features.to(device), test_targets.to(device)

# Create the model and move to the appropriate device
model = create_model(input_size, hidden_size, num_layers, dropout_rate, output_size).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()  # Since we are predicting continuous values
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.001, verbose=True)

# Create DataLoader for batching training data
train_dataset = TensorDataset(train_features, train_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Optionally, create DataLoader for testing data
test_dataset = TensorDataset(test_features, test_targets)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Configuration parameters for gradient clipping
clip_value = 1.0  # Gradient norm threshold



# EarlyStopping class definition
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Instantiate EarlyStopping object
early_stopping = EarlyStopping(patience=20, min_delta=0.001)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients: gradients are modified in place
        clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        running_loss += loss.item()
    
    # Evaluate on test set at the end of each epoch
    if (epoch+1) % 1 == 0:
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()

        # Step the ReduceLROnPlateau scheduler
        scheduler.step(test_loss/len(test_loader))

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}')

    # Early stopping logic
    early_stopping(test_loss/len(test_loader))
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break  # Stop training if early stopping criteria are met

# Save the model after training
torch.save(model.state_dict(), 'lstm_model.pth')
print("Model saved to 'lstm_model.pth'")

# Load the model later (if needed)
# model.load_state_dict(torch.load('lstm_model.pth'))
# model.eval()

# Make predictions on the test set
model.eval()
with torch.no_grad():
    predicted_targets = model(test_features)

# Inverse transform the predicted and actual targets to get original values
predicted_targets = predicted_targets.cpu().numpy()
predicted_targets = target_scaler.inverse_transform(predicted_targets)

actual_targets = test_targets.cpu().numpy()
actual_targets = target_scaler.inverse_transform(actual_targets)

# Visualize the comparison of predicted vs actual target values for a sample in the test set
time_series_sample = test_features.cpu().numpy()[0]  # Take the first sample's time series data

print(predicted_targets[:5])  # Print the raw model output before scaling

# Plot the time series and the predicted vs actual target
# Assuming you already have 'predicted_targets' and 'actual_targets' as numpy arrays after inverse transformation
# Adjust how much of the test set you want to see (first 100 samples, for example)
n_samples = 100

# Create a plot for a subset of the test set (first n_samples)
plt.figure(figsize=(10, 6))

# Plot predicted vs actual for the first n_samples
plt.plot(range(n_samples), predicted_targets[:n_samples], label='Predicted Target', color='red', linestyle='--')
plt.plot(range(n_samples), actual_targets[:n_samples], label='Actual Target', color='green')

# Add titles and labels
plt.legend()
plt.title(f'Predicted vs Actual Target for First {n_samples} Test Samples')
plt.xlabel('Test Sample Index')
plt.ylabel('Target Value')

# Show the plot
plt.show()

# Optionally: Plot the residuals (difference between predicted and actual)
residuals = predicted_targets[:n_samples] - actual_targets[:n_samples]

plt.figure(figsize=(10, 6))
plt.plot(range(n_samples), residuals, label='Residuals (Predicted - Actual)', color='blue')
plt.axhline(0, color='black', linestyle='--')
plt.title(f'Residuals for First {n_samples} Test Samples')
plt.xlabel('Test Sample Index')  
plt.ylabel('Residual (Error)')
plt.legend()
plt.show()