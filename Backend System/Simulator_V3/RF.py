import pandas as pd
import requests
import pickle
import base64


def prepare_data(data):

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

    # Feature: Price Rate of Change
    data['price_roc'] = data['Close'].pct_change()

    # Feature Interaction: Volume-Price Interaction
    data['volume_price_interaction'] = data['Volume'] * data['Close']

    data = data.dropna()

    data['Open - Close'] = data['Open'] - data['Close']
    data['High - Low'] = data['High'] - data['Low']
    data = data.dropna()
    data['Close Tomorrow'] = (data['Close'].shift(-1) > data['Close']).astype(int)* 2 - 1  # * 2 - 1 ==>   replaces 0 with -1

    print(".", end="")

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


    # data = data.dropna(subset=data.columns[data.columns != "Tomorrow"])
    data = data.dropna()
    print(".")
    return data


# Request the train_data and test_data from the server
response = requests.get('http://127.0.0.1:5000/get_data')

# Check if the response status code is OK (200) and the content is not empty
if response.status_code == 200 and response.content:
    try:
        data = response.json()

        # Decode base64 strings and unpickle DataFrames
        train_data = pickle.loads(base64.b64decode(data['train_data']))
        test_data = pickle.loads(base64.b64decode(data['test_data']))

        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
        # train_data = train_data.set_index('Date', drop=True)
        # train_data.to_csv("gg.csv")
        # print(test_data.head())



        #__________________________________________________________________________________________________________
        #algorithm goes here
        if not isinstance(train_data.index, pd.DatetimeIndex):
            train_data = train_data.set_index(pd.to_datetime(train_data.index, utc=True))
        
        #prepare data
        print("preparing train data.", end="")
        train_data = prepare_data(train_data.copy())
        print(train_data.head())

        print("preparing test data.", end="")
        print(test_data)
        test_data = prepare_data(test_data.copy())
        print(test_data.head())

        #splitting data
        X_train = train_data[['Volume', 'Open - Close', 'High - Low', 'Close_Ratio_2', 'Close_Ratio_5', 'Close_Ratio_60', "Close_Ratio_250", 'Trend_2', 'Trend_5', 'Trend_60','trend_Positive_2','trend_Positive_5', 'trend_Positive_60', "sma_20", "ema_12", "roc", "upper_band", "lower_band", "historical_volatility", "price_roc", "volume_price_interaction"]]
        y_train = train_data['Close Tomorrow']

        X_test = test_data[['Volume', 'Open - Close', 'High - Low', 'Close_Ratio_2', 'Close_Ratio_5', 'Close_Ratio_60', "Close_Ratio_250", 'Trend_2', 'Trend_5', 'Trend_60','trend_Positive_2','trend_Positive_5', 'trend_Positive_60', "sma_20", "ema_12", "roc", "upper_band", "lower_band", "historical_volatility", "price_roc", "volume_price_interaction"]]
        y_test = test_data['Close Tomorrow']

        # from sklearn.neighbors import KNeighborsClassifier
        # from sklearn import neighbors
        # from sklearn.model_selection import GridSearchCV #for selecting best parameter (K) for the model
        from sklearn.metrics import accuracy_score, confusion_matrix

        # params = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
        # knn = neighbors.KNeighborsClassifier()
        # model = GridSearchCV(knn, params, cv=5)

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=240, min_samples_split=138, random_state=1)

        #fit the model
        print("training model...")
        model.fit(X_train, y_train)
        
        
        predicted_results = model.predict(X_test)
        print(X_test.head(3))
        results_df = pd.DataFrame({'Date': X_test.index, 'Predicted_Result': predicted_results})
        # results_df.set_index('Date', inplace=True)

        #__________________________________________________________________________________________________________
        model_accurecy = ""
        #Accuracy Score
        accuracy_train = accuracy_score(y_train, model.predict(X_train))
        accuracy_test = accuracy_score(y_test, model.predict(X_test))

        model_accurecy+= f'Train_data Accuracy: {accuracy_train:.2f}%\n'
        model_accurecy+=f'Test_data Accuracy: {accuracy_test:.2f}%\n'

        # Confusion Matrix
        conf_matrix_train = confusion_matrix(y_train, model.predict(X_train))
        conf_matrix_test = confusion_matrix(y_test, model.predict(X_test))

        model_accurecy+="Confusion Matrix for Training Data:"
        model_accurecy+=f"{conf_matrix_train}\n"

        model_accurecy+="Confusion Matrix for Test Data:"
        model_accurecy+= f"{conf_matrix_test}\n"

        # Number of correct predictions for each class
        correct_predictions_1_train = conf_matrix_train[1, 1]
        correct_predictions_minus1_train = conf_matrix_train[0, 0]

        correct_predictions_1_test = conf_matrix_test[1, 1]
        correct_predictions_minus1_test = conf_matrix_test[0, 0]

        model_accurecy+=f'Correct predictions for class "1" in Training Data: {correct_predictions_1_train}\n'
        model_accurecy+=f'Correct predictions for class "-1" in Training Data: {correct_predictions_minus1_train}\n'

        model_accurecy+= f'Correct predictions for class "1" in Test Data: {correct_predictions_1_test}\n'
        model_accurecy+= f'Correct predictions for class "-1" in Test Data: {correct_predictions_minus1_test}\n\n'


        # Serialize the results DataFrame to bytes using Pickle
        serialized_results = base64.b64encode(pickle.dumps(results_df)).decode('utf-8')

        # Prepare the data to be sent as form data
        data_to_send = {
            'model_accurecy': model_accurecy,
            'Prediction': serialized_results
        }

        # Send the results DataFrame back to the server
        response = requests.post('http://127.0.0.1:5000/receive_results', data=data_to_send)

        print(response.text)

    except Exception as e:
        print(f"Error: {str(e)}")
else:
    print(f"Error: {response.status_code} - {response.text}")
