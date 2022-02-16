
# Import packages
import os
import sys
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt




# Data Exploration



# Set date timestamps for historical data download
start_date = datetime.datetime(2010, 1, 1).date()
end_date = datetime.datetime.now().date()
start_date, end_date

print(start_date, end_date)



end_date - start_date

print(end_date - start_date)



ticker = "GOOGL"
# ticker = "MSFT"
# ticker = "FB"


msft = yf.Ticker(ticker)

msft



historical_data = msft.history(start=start_date, end=end_date, interval="1d").reset_index()

historical_data.shape



historical_data.head()



# The above data shows stock prices on per day basis.
#
# + High: Max stock price on that particular day
# + Low: Lowest price went on that particular day
# + Open: Stock price opening on that particular day
# + Close: Final closing price of the stock on that particular day
# + Volume: Amount of stock traded on that day
# + Dividends: Dividens given (if any)
# + Stock Splits: Stock split happend on that particular day (if any)



historical_data.describe()


fig = plt.figure()

plt.plot(historical_data.Close)

plt.legend(["Close", "Open"])



# Feature Engineering



historical_data.head()



historical_data.drop(
    columns=["Dividends", "Stock Splits", "Volume"], inplace=True)



# Add placeholder for forecast
present_date = historical_data.Date.max()
day_number = pd.to_datetime(present_date).isoweekday()

if day_number in [5, 6]:
    next_date = present_date + datetime.timedelta(days=(7-day_number) + 1)
else:
    next_date = present_date + datetime.timedelta(days=1)
print("Present date:", present_date)
print("Next valid date:", next_date)

test_row = pd.DataFrame([[next_date, 0.0, 0.0, 0.0, 0.0]], columns=historical_data.columns)
test_row.head()



historical_data = pd.concat([historical_data, test_row])


# Create lag features
for i in range(1, 7):
    historical_data[f"Close_lag_{i}"] = historical_data.Close.shift(periods=i, axis=0)
    historical_data[f"Open_lag_{i}"] = historical_data.Open.shift(periods=i, axis=0)
    historical_data[f"High_lag_{i}"] = historical_data.High.shift(periods=i, axis=0)
    historical_data[f"Low_lag_{i}"] = historical_data.Low.shift(periods=i, axis=0)

historical_data.head()



historical_data.fillna(0, inplace=True)
historical_data.head()



historical_data.drop(
    columns=["Open", "High", "Low"], inplace=True)



# # Create a holiday dataframe
# min_date, max_date = historical_data.Date.min(), historical_data.Date.max()
# print("Min/Max dates:", min_date, max_date)

# date_range = pd.date_range(start=min_date, end=max_date)
# print("Date range:", date_range.min(), date_range.max())

# # Find dates not present in historical df - closed stock market holiday
# holiday_date_range = [d.date() for d in date_range if d not in historical_data.Date]
# holiday_date_range[:5]


# holidays = pd.DataFrame({
#     "holiday": "shutdown",
#     "ds": pd.to_datetime(holiday_date_range),
#     "lower_bound": 0,
#     "upper_bound": 1,
# })

# holidays.head()



# Modelling



## Facebook's Prophet: Single Timestep Forecasting

# References:
# + https://facebook.github.io/prophet/



import fbprophet as prophet



### Train and Forecast



lag_features = [col for col in historical_data.columns if "lag" in col]



model = prophet.Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode="additive")
for name in lag_features:
    model.add_regressor(name)

model.fit(df=historical_data.iloc[:-1, :].rename(columns={"Date": "ds", "Close":"y"}))



forecast = model.predict(
    historical_data.iloc[-1:][[col for col in historical_data.columns if col != "Close"]].rename(columns={"Date": "ds"})
)

forecast.shape



forecast.yhat


