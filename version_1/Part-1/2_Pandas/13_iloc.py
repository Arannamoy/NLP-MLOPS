# iloc params are only integer
import pandas as pd
data=pd.read_json("data.json")

print(data.iloc[0]) # working for row but not for index. iloc for non label based index


df=pd.DataFrame([1,2,3,4],index=["A","B","C","D"])

# print(df.iloc["A"]) # it gives error


print(data.iloc[5,3]) # for specific cell

# print(data.iloc[5,"Profession"]) # it gives error

