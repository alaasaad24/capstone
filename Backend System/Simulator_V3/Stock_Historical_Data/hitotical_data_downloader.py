import yfinance as yf
import os
import pandas as pd

symbol = "INTC"  #enter any stock symbol

if os.path.exists(f"{symbol}.csv"):
    print(f"Data already exist for {symbol} stock!!!")
else: #download data if not found
    data = yf.Ticker(f"{symbol}")
    data = data.history(period="max")
    data.to_csv(f"{symbol}.csv")