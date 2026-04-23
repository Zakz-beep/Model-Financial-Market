import yfinance as yf
import numpy as np
import pandas as pd

def calculate_close_to_close_volatility(df):
    """
    Menghitung Annualized Realized Volatility menggunakan metode Close-to-Close.
    Membutuhkan DataFrame pandas yang memiliki kolom 'Close'.
    """
    # 1. Hitung Log Return Harian
    # Mengapa log return? Karena log return bersifat simetris dan aditif 
    # untuk perhitungan compounding jangka panjang.
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Hitung Standar Deviasi Harian
    # Ini adalah volatilitas untuk 1 hari
    daily_volatility = log_returns.std()
    
    # 3. Disetahunkan (Annualized)
    # Market biasanya buka ~252 hari dalam setahun.
    # Ingat aturan akar kuadrat waktu: Volatilitas berskala dengan akar waktu.
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    return annualized_volatility

if __name__ == "__main__":
    # Mari kita uji dengan data E-mini S&P 500 Futures (ES=F) selama 1 tahun terakhir
    ticker_symbol = "ES=F"
    print(f"Mengunduh data untuk {ticker_symbol}...")
    
    # Ambil data 1 tahun terakhir
    data = yf.Ticker(ticker_symbol).history(period="1y")
    
    if not data.empty:
        # Panggil fungsi
        realized_vol = calculate_close_to_close_volatility(data)
        
        print("-" * 40)
        print(f"Hasil Kalkulasi Volatilitas Close-to-Close")
        print(f"Aset   : {ticker_symbol}")
        print(f"Periode: 1 Tahun (252 Hari Perdagangan)")
        print(f"Nilai  : {realized_vol * 100:.2f}%")
        print("-" * 40)
    else:
        print("Gagal mengambil data. Periksa koneksi internet Anda.")