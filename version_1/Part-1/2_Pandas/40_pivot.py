import pandas as pd



df=pd.DataFrame({
    "Name":["A","B","C","D"],
    "DSA":[80,80,80,80],
    "OOP":[90,90,90,90],
    "ML":[100,100,100,100]
})

print(df)
df1=df.melt(id_vars="Name",value_name="Score",value_vars=["DSA","OOP","ML"],var_name="Subject")
print(df1)


# pivot is just the reverse process of melt. melt through error if duplicates in df.
print(df1.pivot(index="Name",columns="Subject",values="Score"))