import pandas as pd

data=pd.read_json("data.json")

print(data[(data["Age"]>30) & (data["City"]=="Delhi")]) # filter age>30 and city=Delhi

print(data[(data["Age"]>30) | (data["City"]=="Chennai")]) # filter age>30 or city=Delhi