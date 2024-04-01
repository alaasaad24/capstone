#server libraries
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

#importing models libraries
import joblib
import pandas as pd
import warnings
from datetime import datetime


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def days_difference(start_date, end_date):
    try:
        # Extracting date part from datetime strings
        start_date = start_date.split()[0]
        end_date = end_date.split()[0]
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        difference = abs((end - start).days)
        return difference
    except ValueError:
        return None

# Defining functionds
def MyPredict(symbol, date):
    #choose model
    algo="KNN" 

    #import model
    daily_model = joblib.load(f"./saved models/{algo}_{symbol}_model_d")

    # Load prepared data
    today = pd.read_csv(f"prepared historical data/{symbol}_train_data.csv")

    # Extract the latest date
    today_date = today.iloc[-1].Date

    #predict after how many days
    difference = int(days_difference(today_date, date))

    # Select needed features
    if algo == "KNN":
        today = today[['Volume', 'Open - Close', 'High - Low', 'Close_Ratio_2', 'Close_Ratio_5', 'Close_Ratio_60', "Close_Ratio_250", 'Trend_2', 'Trend_5','trend_Positive_2','trend_Positive_5', "sma_20", "ema_12", "roc", "upper_band", "lower_band", "historical_volatility", "price_roc", "volume_price_interaction"]]
    elif algo == "RF":
        today = today[['Volume', 'Open - Close', 'High - Low', 'Close_Ratio_2', 'Close_Ratio_5', 'Close_Ratio_60', "Close_Ratio_250", 'Trend_2', 'Trend_5', 'Trend_60','trend_Positive_2','trend_Positive_5', 'trend_Positive_60', "sma_20", "ema_12", "roc", "upper_band", "lower_band", "historical_volatility", "price_roc", "volume_price_interaction"]]

    today = today.iloc[-1]
    # print(today)
    warnings.filterwarnings("ignore")
    next_close_prediction = daily_model.predict([today])[0]
    
    message = f"predicted in {str(today_date)} for symbol: {symbol} after {difference} days"
    #return the prediction and message
    return int(next_close_prediction), message


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_json', methods=['POST'])
def process_json():
    #recive
    data = request.json
    print("Received JSON:", data)

     # Check if the symbol is supported
    supported_symbols = ['^IXIC', 'AAPL', 'AMZN', 'GOOGL', 'IBM', 'INTC', 'META', 'MSFT', 'NVDA', 'TSLA']
    symbol = data.get('symbol')

    if symbol not in supported_symbols:
        return jsonify({'message': 'Symbol is not supported!'}), 404

    # Make prediction
    prediction, message = MyPredict(symbol, data.get('date'))
    


    #response
    response_data = {'prediction': prediction, 'message': message}
    print("Response JSON:", response_data, "\n")

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5055, threaded=True)