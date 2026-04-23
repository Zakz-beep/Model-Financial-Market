import numpy as np
import pandas as pd
import yfinance as yf

def hitung_hfrv(ticker, period='60d', interval='5m'):
    
    # Download data 5 menit
    data = yf.download(ticker, period=period, interval=interval)
    
    # Hitung log return setiap 5 menit
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()
    
    # Kuadratkan return
    data['returns_sq'] = data['returns'] ** 2
    
    # Group per hari → jumlahkan return kuadrat
    data.index = pd.to_datetime(data.index)
    rv_daily = data.groupby(data.index.date)['returns_sq'].sum()
    
    # Annualized RV per hari
    rv_daily_ann = np.sqrt(rv_daily * 252) * 100
    
    # RV30 → rata-rata 30 hari terakhir
    rv30 = rv_daily_ann.rolling(30).mean()
    
    return rv_daily_ann, rv30

# Jalankan untuk SPY
rv_daily, rv30 = hitung_hfrv('SPY')

# Ambil VIX untuk hitung VRP
vix = yf.download('^VIX', period='60d', progress=False)['Close']
# Cast explicitly to float, since recent yfinance versions might return a DataFrame/Series slice
vix_today = float(np.squeeze(vix.iloc[-1]))
rv30_today = float(np.squeeze(rv30.iloc[-1]))

vrp = vix_today - rv30_today

print(f"VIX sekarang  : {vix_today:.2f}%")
print(f"RV30 SPY      : {rv30_today:.2f}%")
print(f"VRP           : {vrp:.2f}%")

if vrp > 4.0:
    print("[YES]  Edge bagus untuk short vol")
elif vrp > 0.0:
    print("[WARN] Edge tipis, hati-hati sizing")
else:
    print("[NO]   Hindari short vol sekarang")