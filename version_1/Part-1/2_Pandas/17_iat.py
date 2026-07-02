import pandas as pd

data=pd.read_json("data.json")


print(data.iat[0,0])