import joblib
import pandas as pd


symbol = 'MSFT'

#export model
# joblib.dump(model, f"../saved models/{symbol}_model")



#import model
daily_model = joblib.load(f"./saved models/{symbol}_daily_model")



today = pd.read_csv(f"prepared historical data/{symbol}_train_data.csv")
today_date = today.iloc[-1].Date

today = today[['Volume', 'Open - Close', 'High - Low', 'Close_Ratio_2', 'Close_Ratio_5', 'Close_Ratio_60', "Close_Ratio_250", 'Trend_2', 'Trend_5', 'Trend_60', "Trend_250"]]
today = today.iloc[-1]
print(today)

next_close_prediction = int(daily_model.predict([today])[0])
print(next_close_prediction)
