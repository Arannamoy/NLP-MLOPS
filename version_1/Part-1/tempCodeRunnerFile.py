eth = td.time_series(
    symbol="ETH/USD",
    interval="1day",
    
    timezone="America/New_York",
    start_date="2010-1-1",
    outputsize=365*10,
)
eur.as_pandas().to_csv("./Sample_Data/Daily_ETH_USD_2010_2025.csv")