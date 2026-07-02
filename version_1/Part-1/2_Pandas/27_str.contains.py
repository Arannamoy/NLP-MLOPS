import pandas as pd


df=pd.read_json("data.json")

print(df["City"].str.contains("Kanpur")) # default case sensitive

print(df["City"].str.contains("kanpur",case=False)) # for insensitive