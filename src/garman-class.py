import yfinance as yf
import numpy as np
import pandas as pd

def get_rolling_vol(ticker_symbol, window=5):
    # 1. Ambil data 1 bulan agar punya cukup data untuk perhitungan rolling
    print(f"Mengambil data {ticker_symbol}...")
    df = yf.Ticker(ticker_symbol).history(period="1mo")
    
    if df.empty:
        return None

    # --- Perhitungan Close-to-Close (Rolling) ---
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    # Rolling standard deviation, lalu disetahunkan
    df['Vol_C2C'] = log_returns.rolling(window=window).std() * np.sqrt(252) * 100

    # --- Perhitungan Garman-Klass (Rolling) ---
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    # Rolling mean dari variance, diakar, lalu disetahunkan
    df['Vol_GK'] = np.sqrt(gk_var.rolling(window=window).mean() * 252) * 100

    # --- Hitung Perubahan Harian (Daily Change) ---
    df['Change_C2C'] = df['Vol_C2C'].diff()
    
    # Ambil hanya 7 hari perdagangan terakhir
    last_7_days = df.tail(7).copy()
    
    return last_7_days

def display_report(ticker):
    result = get_rolling_vol(ticker)
    
    if result is not None:
        print(f"\n===== LAPORAN VOLATILITAS 7 HARI TERAKHIR: {ticker} =====")
        print(f"{'Tanggal':<12} | {'C2C Vol (%)':<12} | {'GK Vol (%)':<12} | {'Perubahan (pts)':<15}")
        print("-" * 65)
        
        for index, row in result.iterrows():
            date_str = index.strftime('%Y-%m-%d')
            change_str = f"{row['Change_C2C']:+.2f}" if not np.isnan(row['Change_C2C']) else "N/A"
            print(f"{date_str:<12} | {row['Vol_C2C']:>11.2f}% | {row['Vol_GK']:>11.2f}% | {change_str:>14}")
        
        print("-" * 65)
    else:
        print(f"Gagal memproses data untuk {ticker}")

if __name__ == "__main__":
    # Bandingkan ES Futures dan Gold Futures
    display_report("ES=F")
    display_report("GC=F")