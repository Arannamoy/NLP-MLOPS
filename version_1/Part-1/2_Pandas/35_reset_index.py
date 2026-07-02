import pandas as pd


df=pd.read_json("data.json")

df1=df.copy()

df1=df1.sort_values(["Age","Name"])

print(df1.reset_index()) # it shows previous index

print(df1.reset_index(drop=True)) # it does not show previous index only shows current index. it returns views not copy

print(df1.reset_index(drop=True,inplace=True)) # but it returns copy

print(df1)