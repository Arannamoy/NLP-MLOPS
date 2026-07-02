import pandas as pd


data=[
    {"Name":"Hello","Age":"25","City":"NY"},
    {"Name":"Hi","Age":"25","City":"San Jose"}
]


print(pd.json_normalize(data))