import pandas as pd


df=pd.read_json("data.json")

df1=df.copy()

print(df1.sort_index()) # sort based on index