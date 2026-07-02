import pandas as pd


df=pd.read_json("data.json")

df1=df.copy()


print(df1.sort_values("Age")) 

print(df1.sort_values(["Age"])) # if two rows same then for next comparison. df.sort_values(["Age","Salary"])

