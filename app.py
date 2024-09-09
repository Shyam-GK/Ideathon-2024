from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Function to wait if needed to respect API rate limits
def wait_if_needed(last_request_time, rate_limit_interval=12):
    current_time = time.time()
    elapsed_time = current_time - last_request_time
    if elapsed_time < rate_limit_interval:
        wait_time = rate_limit_interval - elapsed_time
        print(f"Waiting for {wait_time:.2f} seconds to respect API rate limits...")
        time.sleep(wait_time)
    return time.time()

# Load data from Yahoo Finance API (stocks)
def load_stock_data(symbol, last_request_time):
    try:
        last_request_time = wait_if_needed(last_request_time)
        print(f"Loading stock data for {symbol}...")
        df = yf.download(symbol, period='max', interval='1d')
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        df = df[['Close']].rename(columns={'Close': 'Close'})
        return df, last_request_time
    except Exception as e:
        print(f"Error loading stock data for {symbol}: {e}")
        return pd.DataFrame(), last_request_time

# Load data from Yahoo Finance API (cryptocurrency)
def load_crypto_data(symbol, last_request_time):
    try:
        last_request_time = wait_if_needed(last_request_time)
        print(f"Loading cryptocurrency data for {symbol}...")
        df = yf.download(symbol, period='max', interval='1d')
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        df = df[['Close']].rename(columns={'Close': 'Close'})
        return df, last_request_time
    except Exception as e:
        print(f"Error loading cryptocurrency data for {symbol}: {e}")
        return pd.DataFrame(), last_request_time

# Preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# Create and train model
def create_and_train_model(x_train, y_train, model_path):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(x_train, y_train, batch_size=1, epochs=1)  # Increased epochs for better training
    model.save(model_path)
    return model

# Prepare data for training
def prepare_training_data(scaled_data, investment_period):
    x_train, y_train = [], []
    for i in range(investment_period, len(scaled_data)):
        x_train.append(scaled_data[i-investment_period:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train

# Predict future values
def predict_future_values(model, scaler, df, investment_period):
    test_data = df[['Close']].tail(investment_period).values
    scaled_test_data = scaler.transform(test_data)
    x_test = np.array([scaled_test_data])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions[0, 0]

# Get or create model
def get_model(x_train, y_train, model_path):
    try:
        model = load_model(model_path)
        print(f"Model {model_path} loaded successfully.")
    except Exception as e:
        print(f"Model not found or could not be loaded. Creating a new one. Error: {e}")
        model = create_and_train_model(x_train, y_train, model_path)
    return model

# Convert investment period to days
def convert_to_days(period, unit):
    if unit == 'days':
        return period
    elif unit == 'months':
        return period * 30  # Assuming 1 month = 30 days
    elif unit == 'years':
        return period * 365  # Assuming 1 year = 365 days
    else:
        return period  # Fallback to days if something goes wrong

# Generate and save graphs using Matplotlib
def generate_graph(symbol, df):
    if df.empty:
        print(f"No data to plot for {symbol}")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label=f'{symbol} Close Price')
    plt.title(f'{symbol} Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.legend(loc='best')
    
    # Ensure the 'static/graphs' directory exists
    if not os.path.exists('static/graphs'):
        os.makedirs('static/graphs')
        
    graph_path = f'static/graphs/{symbol}.png'
    plt.savefig(graph_path)
    plt.close()

# Home route to render the form
@app.route('/')
def home():
    return render_template('index.html')  # Render the form on the home page

# Route to handle the form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name')
    age = int(request.form.get('age'))
    investment_amount = float(request.form.get('investment_amount'))
    investment_period = int(request.form.get('investment_period'))
    period_unit = request.form.get('period_unit')  # Unit: days, months, or years

    # Convert investment period to days
    investment_period_in_days = convert_to_days(investment_period, period_unit)

    print(f"Received input: Name={name}, Age={age}, Investment Amount={investment_amount}, Investment Period={investment_period_in_days} days")

    # List of assets including stocks, cryptocurrencies, and gold
    symbols = {
        'MSFT': 'stock',
        'AAPL': 'stock',
        'GOOGL': 'stock',
        'TSLA': 'stock',
        'JPM': 'stock',
        'GS': 'stock',
        'MS': 'stock',
        'WMT': 'stock',
        'CAT': 'stock',
        'BTC-USD': 'crypto',
        'ETH-USD': 'crypto',
        'GC=F': 'gold'  # Gold Futures
    }

    last_request_time = time.time()
    dfs = {}

    for symbol, asset_type in symbols.items():
        if asset_type == 'stock':
            df, last_request_time = load_stock_data(symbol, last_request_time)
        elif asset_type == 'crypto':
            df, last_request_time = load_crypto_data(symbol, last_request_time)
        elif asset_type == 'gold':
            df, last_request_time = load_stock_data(symbol, last_request_time)  # Gold is treated similarly to stocks for data fetching
        
        if df.empty:
            print(f"No data for {symbol}")
            continue
        dfs[symbol] = df

        # Generate graph for each symbol
        generate_graph(symbol, df)

    scalers = {}
    models = {}
    predicted_returns = {}

    for symbol, df in dfs.items():
        scaled_data, scaler = preprocess_data(df)
        x_train, y_train = prepare_training_data(scaled_data, investment_period_in_days)
        model_path = f'model_{symbol}.h5'
        
        # Get or create and save model
        models[symbol] = get_model(x_train, y_train, model_path)
        
        scalers[symbol] = scaler

        future_value = predict_future_values(models[symbol], scalers[symbol], df, investment_period_in_days)
        predicted_returns[symbol] = future_value

    print(f"Predicted returns: {predicted_returns}")

    # After predicted_returns dictionary is filled, get the best result
    best_symbol = max(predicted_returns, key=predicted_returns.get)
    best_return = predicted_returns[best_symbol]

    print(f"Best prediction: {best_symbol} with returns of ${best_return:.2f}")

    return render_template('result.html', 
                           name=name, 
                           investment_amount=investment_amount, 
                           investment_period=investment_period_in_days, 
                           predicted_returns=predicted_returns,
                           best_symbol=best_symbol, 
                           best_return=best_return)

# Route to display graphs
@app.route('/view-graphs')
def view_graphs():
    symbols = ['MSFT', 'AAPL', 'GOOGL', 'TSLA', 'JPM', 'GS', 'MS', 'WMT', 'CAT', 'BTC-USD', 'ETH-USD', 'GC=F']
    return render_template('view_graphs.html', symbols=symbols)

# Route to show a specific graph
@app.route('/show-graph/<symbol>')
def show_graph(symbol):
    return render_template('show_graph.html', symbol=symbol)

if __name__ == '__main__':
    # Ensure the folder exists to save graphs
    if not os.path.exists('static/graphs'):
        os.makedirs('static/graphs')
        
    app.run(debug=True)
