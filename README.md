Stock Price Prediction Using LSTM

### Introduction

This repository contains a collection of scripts developed collaboratively on Git for predicting stock prices using an LSTM model. The scripts are designed to handle data preprocessing, model training, and predictions effectively. This project aims to explore the potential of neural networks in predicting day-to-day stock price movements, with an emphasis on understanding the limitations of such models in trading environments.

### Repository Structure

data_processing.py: Handles the initial data preprocessing, including cleaning and feature engineering. This script prepares the raw stock data by adjusting for splits, calculating necessary financial indicators, and normalizing the data.
data_loader.py: Contains the functionality to load data into a format suitable for training the LSTM model. This includes splitting the data into training and test sets.
model.py: Defines the LSTM architecture, incorporating techniques such as batch normalization, gradient clipping, and an adaptive learning rate to enhance training stability and performance.
master.py: Orchestrates the process of training the model on the preprocessed data, managing the flow from data loading, model training, to saving the trained model.
LSTM Architecture

The model is built using an LSTM (Long Short-Term Memory) architecture, which is particularly suited for time-series prediction due to its ability to capture long-term dependencies in sequence data. Our LSTM model consists of several layers optimized for time series forecasting:

Input Layer: Takes in sequences of stock prices along with other financial indicators.
LSTM Layers: Processes the input data, with each layer providing a higher level of abstraction.
Batch Normalization: Applied within the LSTM layers to normalize activations, thus improving the training speed and stability.
Output Layer: Produces the prediction for the next day's stock price movement.
Features and Techniques
Gradient Clipping: Used to prevent exploding gradients, thereby ensuring stable training across batches.
Adaptive Learning Rate: Implemented through the optimizer to adjust the learning rate based on training progress, enhancing the model's ability to converge to optimal weights.
Batch Normalization: Helps in normalizing the input layer by adjusting and scaling activations.
Challenges and Conclusions

One significant challenge encountered in this project is the inherent randomness in day trading. Despite incorporating various sophisticated features and financial indicators, our conclusions strongly suggest that predicting day-to-day stock movements is extremely challenging and, at times, too random to forecast reliably with the given features. This highlights the unpredictable nature of financial markets and the limitations of using historical data for future price predictions.

### Setup and Usage

Clone the repository to your local machine.
Ensure you have Python and necessary libraries installed (pandas, numpy, matplotlib, tensorflow, etc.).
Run master.py to execute the full pipeline from data processing to model training.
Check the outputs in the designated output directories.
Collaborators

This project was a collaborative effort, developed with a student-friend of mine through team contributions and version control.

