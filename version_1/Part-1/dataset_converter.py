import os
import pandas as pd

files=os.listdir("Sample_Data")
print(files)

for file in files:
    name=file.split(".")
    df=pd.read_csv(f"./Sample_Data/{name[0]}.csv")
    df.to_excel(f"./Sample_Data/{name[0]}.xlsx")
    df.to_json(f"./Sample_Data/{name[0]}.json")
    print(name)