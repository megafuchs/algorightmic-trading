import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Data/tech_indices.csv', skiprows=4)
df.drop(columns=['Openint'], inplace=True)

# Define split dates and prepare for annotations
split_dates = {
    '2000-06-21': '2-for-1 Split',
    '2005-02-28': '2-for-1 Split',
    '2014-06-09': '7-for-1 Split',
    '2020-08-31': '4-for-1 Split'
}

# Plotting adjusted close prices
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'], label='Adjusted Close Prices', color='blue')
for date, label in split_dates.items():
    split_date = pd.to_datetime(date)
    plt.axvline(x=split_date, color='red', linestyle='--', linewidth=1)
    plt.text(split_date, plt.ylim()[1], f'{label}\n{date}', horizontalalignment='right', color='red')
plt.title('AAPL Adjusted Close Prices with Stock Splits Indicated')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()

# Calculate target
def create_target_percentage(df):
    df['Target'] = ((df['Close'].shift(-1) - df['Close']) / df['Close'] * 300).round(4)
    df.dropna(subset=['Target'], inplace=True)
    return df
df = create_target_percentage(df)

# Calculate indicators
def calculate_rsi(df, periods=14):
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=periods-1, min_periods=periods).mean()
    avg_loss = loss.ewm(com=periods-1, min_periods=periods).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = (100 - (100 / (1 + rs))).round(2)
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = (short_ema - long_ema).round(3)
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean().round(3)
    return df

df = calculate_rsi(df)
df = calculate_macd(df)

# Essential columns
columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line', 'Target']
df = df[columns]

# Clean and validate data
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.dropna(inplace=True)

# Save processed data
df.to_csv('Data/processed_stock_data.csv', index=False)

print("Data processing and visualization complete. Output saved.")
