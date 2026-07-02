import pandas as pd


df=pd.read_json("data.json")

joining_date=df["Joining Date"].copy()


joining_date=joining_date.dropna()


joining_date=pd.to_datetime(joining_date)


print(joining_date)