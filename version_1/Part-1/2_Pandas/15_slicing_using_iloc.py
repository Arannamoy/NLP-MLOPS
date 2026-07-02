import pandas as pd

data=pd.read_json("data.json")

print(data.iloc[0:3,0:3])