# Features of the Investment Guidance Application

## Key Features

### Data Loading
- **Stocks**: Fetches historical stock data using the Yahoo Finance API.
- **Cryptocurrencies**: Retrieves historical cryptocurrency data.
- **Gold**: Retrieves historical data for gold futures.

### Data Preprocessing
- **Scaling**: Uses MinMaxScaler to scale the data to the range [0, 1].
- **Training Data Preparation**: Prepares data sequences for model training.

### Model Training
- **LSTM Model**: Implements an LSTM neural network to predict future values.
- **Model Saving**: Saves trained models to disk for future use.

### Prediction
- **Future Value Prediction**: Predicts future asset prices based on user-defined investment periods.

### Graph Generation
- **Graph Creation**: Generates graphs showing historical closing prices of assets.
- **Graph Display**: Allows users to view generated graphs on the web interface.

### Web Interface
- **Form Input**: Collects user input for name, age, investment amount, and period.
- **Results Display**: Shows predicted returns and the best investment option.
- **Graph Viewing**: Provides options to view graphs for various assets.

## Future Enhancements
- **Support for More Assets**: Include additional cryptocurrencies or commodities.
- **User Authentication**: Implement user login and profile management.
<<<<<<< HEAD
- **Enhanced Visualization**: Improve graph features and interactivity.
=======
- **Enhanced Visualization**: Improve graph features and interactivity.
>>>>>>> ac806044fd19503e34408510cceccab4d041dbd4
