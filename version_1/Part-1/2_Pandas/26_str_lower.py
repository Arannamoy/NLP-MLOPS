import pandas as pd


df=pd.read_json("data.json")

print(df["Name"].str.lower())