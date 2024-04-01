import pandas as pd
import numpy as np

data = pd.read_csv("finantial_report_2y.csv")
data = data.iloc[::-1]
data.to_csv("reversed_financial_report_2y.csv", index=False)
