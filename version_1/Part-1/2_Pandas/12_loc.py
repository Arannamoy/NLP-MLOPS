import pandas as pd
data=pd.read_json("data.json")

print(data.loc[0]) # working for both row and index but iloc not working for both. loc for label based index


df=pd.DataFrame([1,2,3,4],index=["A","B","C","D"])

print(df.loc["A"]) # for column

print(data.loc[5,"Profession"]) # for specific cell

