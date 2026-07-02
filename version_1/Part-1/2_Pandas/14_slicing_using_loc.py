import pandas as pd

data=pd.read_json("data.json")

print(data.loc[0:3,["Name","City","Profession"]])