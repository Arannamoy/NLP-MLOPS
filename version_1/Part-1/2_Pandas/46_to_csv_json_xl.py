import pandas as pd


data=[
    {"Name":"Hello","Age":"25","City":"NY"},
    {"Name":"Hi","Age":"25","City":"San Jose"}
]


df=pd.json_normalize(data)

df.to_csv("to.csv")

df.to_json("to.json")