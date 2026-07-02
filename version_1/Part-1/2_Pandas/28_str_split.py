import pandas as pd


df=pd.read_json("data.json")

print(df["Full Stack"].str.split("+"))