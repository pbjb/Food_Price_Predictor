import pandas as pd
from prophet import Prophet

def forecast_price(item, days, data_path="data/food_prices.csv"):
    df = pd.read_csv(data_path)
    df = df[df["Item"] == item][["Date", "Price"]]
    df.rename(columns={"Date": "ds", "Price": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])
    
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]].tail(days)
