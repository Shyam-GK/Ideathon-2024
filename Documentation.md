# Investment Guidance Application Documentation

## Overview

This Flask application predicts investment returns for stocks, cryptocurrencies, and gold. It uses historical data from Yahoo Finance, preprocesses it, and applies an LSTM (Long Short-Term Memory) neural network model to make future predictions. The application also generates and displays graphs of asset prices.

## Features

- **Data Retrieval**: Fetches historical data for stocks, cryptocurrencies, and gold from Yahoo Finance.
- **Data Preprocessing**: Scales and prepares data for training using MinMaxScaler.
- **Model Training**: Creates and trains an LSTM model to predict future asset prices.
- **Prediction**: Predicts future values based on user-defined investment periods.
- **Graph Generation**: Creates visual graphs of asset prices over time.
- **Web Interface**: Provides a user-friendly interface for input and displays results, including graphs.

## File Structure

- `app.py`: The main Flask application file containing routes and logic for data processing, model training, and predictions.
- `templates/`
  - `index.html`: HTML form for user input.
  - `result.html`: Displays the investment prediction results.
  - `show_graph.html`: Shows individual asset graphs.
  - `view_graphs.html`: Lists buttons to view graphs of different assets.
- `static/`
  - `graphs/`: Directory where generated graphs are saved.

## Dependencies

The application requires the following Python packages:

- Flask
- NumPy
- Pandas
- yfinance
- scikit-learn
- TensorFlow
- Matplotlib

These can be installed via `pip` as shown in the installation instructions.

## Usage

1. Run the Flask application using:
    ```bash
    python app.py
    ```
2. Access the web interface at `http://127.0.0.1:5000/`.
3. Enter the required details into the form and submit to see investment predictions.
4. If the model has not been trained, it will take some time initially. Adjust the number of epochs if needed for more accurate results.

## Contact

For questions or support, please contact Shyam GK at shyamgokulkrish@gmail.com .