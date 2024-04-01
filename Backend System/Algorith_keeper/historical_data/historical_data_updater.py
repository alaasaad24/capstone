import yfinance as yf
import pandas as pd
import os

def update_csv(symbol):
    # File path for the CSV file
    csv_file_path = f"historical_data/{symbol}.csv"

    # Load existing data from CSV
    if os.path.exists(csv_file_path):
        existing_data = pd.read_csv(csv_file_path, index_col="Date", parse_dates=True)
    else:
        data = yf.Ticker(symbol)
        data = data.history(period="max")
        data.to_csv(csv_file_path)
        print(f"Data for {symbol} is downloaded successfully")
        return

    # Fetch latest data from yfinance
    latest_data = yf.Ticker(symbol).history(period="2d")

    # Check if data is not updated for a long time
    old_data = latest_data.index[-2]
    if old_data not in existing_data.index:
        data = yf.Ticker(symbol)
        data = data.history(period="max")
        data.to_csv(csv_file_path)
        print(f"Data for {symbol} is downloaded from the beginning successfully because data was not updated for a long time.")
        return

    # Check if data for the current date already exists in the CSV
    latest_date = latest_data.index[-1]

    if latest_date not in existing_data.index:
        # Concatenate the latest data to the existing data and save to CSV
        updated_data = pd.concat([existing_data, latest_data])
        updated_data.to_csv(csv_file_path)
        print(f"Data for {latest_date.date()} added to {csv_file_path}")
    else:
        print(f"Data for {latest_date.date()} already exists in {csv_file_path}")

# List of symbols to update
symbols_to_update = ["MSFT", "AAPL", "TSLA", "IBM", "^IXIC", "AMZN", "GOOGL", "INTC", "META", "NVDA"]

# Update each symbol
for symbol in symbols_to_update:
    update_csv(symbol)