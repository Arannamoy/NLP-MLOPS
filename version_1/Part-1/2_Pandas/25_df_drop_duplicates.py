import pandas as pd

df=pd.read_json("data.json")

print(df.drop_duplicates()) # remove duplicates from whole dataset

print(df.drop_duplicates(subset=["Age","Name"])) # remove duplicate from based on these columns