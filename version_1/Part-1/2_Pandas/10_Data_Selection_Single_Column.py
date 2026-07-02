import pandas as pd
data=pd.read_json("data.json")

print(data["Name"]) # data[_column_name_] for single column
