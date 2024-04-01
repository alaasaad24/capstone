import os
import pandas as pd
import joblib

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV #for selecting best parameter (K) for the model

from sklearn.ensemble import RandomForestClassifier




symbols = ['^IXIC', 'AAPL', 'AMZN', 'GOOGL', 'IBM', 'INTC', 'META', 'MSFT', 'NVDA', 'TSLA']

for symbol in symbols:

    print(f"{symbol}:\nPreparing data for training\n")
    #prediction
    if os.path.exists(f"historical_data/{symbol}.csv"):
        data = pd.read_csv(f"historical_data/{symbol}.csv", index_col=0)


    # Convert the index to datetime with utc=True
    data.index = pd.to_datetime(data.index, utc=True)

    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index(pd.to_datetime(data.index, utc=True))

    # Filter data to include only records from 1986
    data = data[data.index.year >= 1986]



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

    #splitting data
    X_train = data[['Volume', 'Open - Close', 'High - Low', 'Close_Ratio_2', 'Close_Ratio_5', 'Close_Ratio_60', "Close_Ratio_250", 'Trend_2', 'Trend_5','trend_Positive_2','trend_Positive_5', "sma_20", "ema_12", "roc", "upper_band", "lower_band", "historical_volatility", "price_roc", "volume_price_interaction"]]
    y_train = data['Close Tomorrow']



    print(f"Training the model using KNN Classifier...\n")

    #train the model using KNN
    params = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
    knn = neighbors.KNeighborsClassifier()
    model = GridSearchCV(knn, params, cv=5)

    model.fit(X_train, y_train)
    joblib.dump(model, f"./saved models/KNN_{symbol}_model_d")


    #train the model using Random Forest
    print(f"Training the model using Random Forest Classifier...\n")
    X_train = data[['Volume', 'Open - Close', 'High - Low', 'Close_Ratio_2', 'Close_Ratio_5', 'Close_Ratio_60', "Close_Ratio_250", 'Trend_2', 'Trend_5', 'Trend_60','trend_Positive_2','trend_Positive_5', 'trend_Positive_60', "sma_20", "ema_12", "roc", "upper_band", "lower_band", "historical_volatility", "price_roc", "volume_price_interaction"]]

    model = RandomForestClassifier(n_estimators=240, min_samples_split=138, random_state=1)
    model.fit(X_train, y_train)
    joblib.dump(model, f"./saved models/RF_{symbol}_model_d")
