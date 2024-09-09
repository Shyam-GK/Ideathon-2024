The **Investment Guidance** project is a web application designed to provide investment predictions based on historical stock and cryptocurrency data. It allows users to input their personal details and investment parameters to receive predictions on future investment values. The application utilizes machine learning models to analyze historical data and generate predictions.

## Features

- **Interactive Web Form**: Allows users to input their personal details, investment amount, and investment period to get predictions.
- **Dynamic Background Video**: Enhances user experience with an engaging video background.
- **Machine Learning Predictions**: Utilizes LSTM neural networks to predict future investment values.
- **Data Visualization**: Generates and displays graphs of historical data for stocks, cryptocurrencies, and gold.

## Technologies

- **Frontend**: HTML, CSS
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow (Keras), scikit-learn
- **Data Source**: Yahoo Finance API
- **Data Visualization**: Matplotlib

## Setup and Installation

### Prerequisites

- Python 3.x
- Flask
- TensorFlow
- scikit-learn
- yfinance
- pandas
- numpy
- matplotlib

### Installation Steps

1. *Clone the repository*:

    ```bash
    git clone https://github.com/Shyam-GK/Ideathon-2024.git
    cd Ideathon-2024
    ```
    
2. *Install the required packages*:

    ```bash
    pip install Flask numpy pandas yfinance scikit-learn tensorflow matplotlib
    ```
    

3. *Download or train machine learning models*:

    Ensure that model files present in Models folder are present in the project directory. If they are not, the application will train new models automatically when it runs.

4. *Run the Flask application*:

    ```bash
    python app.py
    ```
    

    The application will be available at http://127.0.0.1:5000/.

## Usage

1. Open a web browser and navigate to http://127.0.0.1:5000/.
2. Fill out the form with your name, age, investment amount, and investment period.
3. Click the "Predict" button to submit the form and receive predictions.
4. View the predicted returns and graphs on the results page.
5. To view generated graphs, navigate to http://127.0.0.1:5000/view-graphs.

## File Structure

- app.py: Main Flask application file that contains backend logic.
- templates/: Directory for HTML templates.
  - index.html: Contains the form for user input.
  - result.html: Displays the prediction results.
  - view_graphs.html: Displays available graphs.
  - show_graph.html: Shows a specific graph.
- static/: Directory for static files.
  - graphs/: Directory where generated graphs are saved.
- requirements.txt: Lists the Python dependencies required for the project.
- Models: Machine learning model files (if they exist).

## Contributing

Contributions to the project are welcome. To contribute, please fork the repository, make changes, and submit a pull request. For reporting issues or suggesting improvements, please use the Issues tab on GitHub.


## Acknowledgments

- [Yahoo Finance API](https://www.yahoofinanceapi.com/) for financial data.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for machine learning.
- [Flask](https://flask.palletsprojects.com/) for the web application framework.
- [Matplotlib](https://matplotlib.org/) for data visualization.

---
