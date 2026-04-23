import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

def hitung_hfrv_daily(ticker='SPY', period='120d', interval='5m'):
    data = yf.download(ticker, period=period, interval=interval)
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['returns_sq'] = data['returns'] ** 2
    data.index = pd.to_datetime(data.index)
    rv_daily = data.groupby(data.index.date)['returns_sq'].sum()
    rv_daily_ann = np.sqrt(rv_daily * 252) * 100
    return rv_daily_ann

def har_model_forecast(rv_series):
    df = pd.DataFrame({'RV': rv_series})
    
    # Buat fitur HAR
    df['RV_d'] = df['RV'].shift(1)           # kemarin
    df['RV_w'] = df['RV'].shift(1).rolling(5).mean()   # 5 hari
    df['RV_m'] = df['RV'].shift(1).rolling(22).mean()  # 22 hari
    df = df.dropna()
    
    # Split train/test
    train = df.iloc[:-5]
    test  = df.iloc[-5:]
    
    # Features & target
    features = ['RV_d', 'RV_w', 'RV_m']
    X_train = train[features]
    y_train = train['RV']
    X_test  = test[features]
    
    # Fit HAR model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Forecast RV besok
    forecast = model.predict(X_test)
    
    return forecast[-1]  # RV forecast untuk hari berikutnya

# Jalankan full pipeline
rv_series = hitung_hfrv_daily('SPY')
forecast_rv = har_model_forecast(rv_series)

# Ambil IV dari VIX
vix = yf.download('^VIX', period='5d')['Close'].iloc[-1]

# Hitung VRP dan generate signal
vrp = vix - forecast_rv
print(f"IV (VIX)       : {vix:.2f}%")
print(f"Forecast RV    : {forecast_rv:.2f}%")
print(f"VRP            : {vrp:.2f}%")

if vrp > 3:
    print("✅ SHORT VOL signal")
elif vrp < -3:
    print("🔴 LONG VOL signal")
else:
    print("⚠️ No trade — edge terlalu tipis")