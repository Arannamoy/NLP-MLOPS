from twelvedata import TDClient
from dotenv import load_dotenv
import os

load_dotenv()
api=os.getenv('api1')

td = TDClient(apikey=api)


# eur = td.time_series(
#     symbol="EUR/USD",
#     interval="1day",
    
#     timezone="America/New_York",
#     start_date="2010-1-1",
#     outputsize=365*10,
# )
# eur.as_pandas().to_csv("./Sample_Data/Daily_EUR_USD_2010_2025.csv")

# xau = td.time_series(
#     symbol="XAU/USD",
#     interval="1day",
    
#     timezone="America/New_York",
#     start_date="2010-1-1",
#     outputsize=365*10,
# )
# xau.as_pandas().to_csv("./Sample_Data/Daily_XAU_USD_2010_2025.csv")

# nvda = td.time_series(
#     symbol="NVDA",
#     interval="1day",
    
#     timezone="America/New_York",
#     start_date="2010-1-1",
#     outputsize=365*10,
# )
# nvda.as_pandas().to_csv("./Sample_Data/Daily_NVDA_2010_2025.csv")


# goog= td.time_series(
#     symbol="GOOG",
#     interval="1day",
#     timezone="America/New_York",
#     start_date="2010-1-1",
#     outputsize=365*10,
# )

# goog.as_pandas().to_csv("./Sample_Data/Daily_GOOG_2010_2025.csv")

# msft = td.time_series(
#     symbol="MSFT",
#     interval="1day",
    
#     timezone="America/New_York",
#     start_date="2010-1-1",
#     outputsize=365*10,
# )
# msft.as_pandas().to_csv("./Sample_Data/Daily_MSFT_2010_2025.csv")


# btc = td.time_series(
#     symbol="BTC/USD",
#     interval="1day",
    
#     timezone="America/New_York",
#     start_date="2010-1-1",
#     outputsize=365*10,
# )
# btc.as_pandas().to_csv("./Sample_Data/Daily_BTC_USD_2010_2025.csv")


# eth = td.time_series(
#     symbol="ETH/USD",
#     interval="1day",
    
#     timezone="America/New_York",
#     start_date="2010-1-1",
#     outputsize=365*10,
# )
# eth.as_pandas().to_csv("./Sample_Data/Daily_ETH_USD_2010_2025.csv")


# sol = td.time_series(
#     symbol="SOL/USD",
#     interval="1day",
    
#     timezone="America/New_York",
#     start_date="2010-1-1",
#     outputsize=365*10,
# )
# sol.as_pandas().to_csv("./Sample_Data/Daily_SOL_USD_2010_2025.csv")

# jpy = td.time_series(
#     symbol="USD/JPY",
#     interval="1day",
    
#     timezone="America/New_York",
#     start_date="2010-1-1",
#     outputsize=365*10,
# )
# jpy.as_pandas().to_csv("./Sample_Data/Daily_JPY_USD_2010_2025.csv")



gbp = td.time_series(
    symbol="GBP/USD",
    interval="1day",
    
    timezone="America/New_York",
    start_date="2010-1-1",
    outputsize=365*10,
)
gbp.as_pandas().to_csv("./Sample_Data/Daily_GBP_USD_2010_2025.csv")