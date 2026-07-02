import pandas as pd


df=pd.read_json("data.json")

df1=df.copy()


df1["Rank"]=df["Age"].rank(ascending=True) # Create a new column based on Age rank and 


print(df1)