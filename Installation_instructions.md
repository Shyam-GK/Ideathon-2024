# Installation Instructions for Investment Guidance Application

## Prerequisites

- **Python**: Version 3.7 or higher is recommended.
- **pip**: Ensure that pip is installed to manage Python packages.

## Installation Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/investment-guidance-app.git
    cd investment-guidance-app
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install Flask numpy pandas yfinance scikit-learn tensorflow matplotlib
    ```

4. **Run the Application**

    ```bash
    python app.py
    ```

5. **Access the Web Interface**

    Open your web browser and go to `http://127.0.0.1:5000/`.

## Additional Notes

- Ensure that you have an active internet connection for data fetching and model training.
- You may need to install additional system libraries depending on your operating system.
