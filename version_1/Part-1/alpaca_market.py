from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
import pandas as pd

client = CryptoHistoricalDataClient()
request_params = CryptoBarsRequest(
  symbol_or_symbols=["BTC/USD","ETH/USD","SOL/USD","XAU/USD","XMR/USD"],
  timeframe=TimeFrame.Minute,
  start=datetime(2021,11,7),
  end=datetime(2025,11,7),
  
)


data = client.get_crypto_bars(request_params)

data.df.to_csv("./Sample_Data/Alpaca_Crypto_Market_Data.csv")

print(data.df)




