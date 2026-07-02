import pandas as pd


df=pd.read_json("data.json")

df2=df.dropna()

print(df2)

df1=df.dropna(axis=1)

print(df1)