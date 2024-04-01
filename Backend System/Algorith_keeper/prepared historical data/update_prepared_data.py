import os
import pandas as pd

symbols = ['^IXIC', 'AAPL', 'AMZN', 'GOOGL', 'IBM', 'INTC', 'META', 'MSFT', 'NVDA', 'TSLA']

for symbol in symbols:

    #prediction
    if os.path.exists(f"historical_data/{symbol}.csv"):
        data = pd.read_csv(f"historical_data/{symbol}.csv", index_col=0)


    # Convert the index to datetime with utc=True
    data.index = pd.to_datetime(data.index, utc=True)

    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index(pd.to_datetime(data.index, utc=True))

    # Filter data to include only records from 2016 onwards
    data = data[data.index.year >= 2022]



    data['Volume'] = data['Volume']/10000000
        # Feature: Moving Averages

    data['sma_20'] = data['Close'].rolling(window=20).mean() # 20-day Simple Moving Average
    data['ema_12'] = data['Close'].ewm(span=12, adjust=False).mean() # 12-day Exponential Moving Average

    # Feature: Rate of Change
    data['roc'] = data['Close'].pct_change() * 100 # Percentage change in closing price

    # Feature: Bollinger Bands
    data['upper_band'], data['lower_band'] = data['Close'].rolling(window=20).mean() + 2 * data['Close'].rolling(window=20).std(), data['Close'].rolling(window=20).mean() - 2 * data['Close'].rolling(window=20).std()

    # Feature: Historical Volatility
    data['historical_volatility'] = data['Close'].pct_change().rolling(window=20).std() * (252 ** 0.5) # Annualized volatility
    data['price_roc'] = data['Close'].pct_change()
    data['volume_price_interaction'] = data['Volume'] * data['Close']

    data = data.dropna()

    data['Open - Close'] = data['Open'] - data['Close']
    data['High - Low'] = data['High'] - data['Low']
    data = data.dropna()
    data['Close Tomorrow'] = (data['Close'].shift(-1) > data['Close']).astype(int)* 2 - 1  # * 2 - 1 ==>   replaces 0 with -1


    horizons = [2,5,60,250] #2days , week, ...

    for horizon in horizons:
        rolling_averages = data.rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Close Tomorrow"]

        # Calculate sum of positive and negative values separately
        positive_sum_column = f"trend_Positive_{horizon}"
        negative_sum_column = f"trend_Negative_{horizon}"

        data[positive_sum_column] = data.shift(1).rolling(horizon).apply(lambda x: (x[x == 1]).sum(), raw=True)["Close Tomorrow"]
        data[negative_sum_column] = data.shift(1).rolling(horizon).apply(lambda x: (x[x == -1]).sum(), raw=True)["Close Tomorrow"]

    data = data.dropna()

    #save the prepared data
    data.to_csv(f"prepared historical data/{symbol}_train_data.csv")
    
    print(f"symbol: {symbol} data prepared sucsessfuly\n")
