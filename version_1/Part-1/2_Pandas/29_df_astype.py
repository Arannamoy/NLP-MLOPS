import pandas as pd



df=pd.read_json("data.json")

df1=df["Age"].fillna(0) # filled NaN with 0

print(df1.astype(float))

print(df1.info())