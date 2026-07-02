import pandas as pd

df=pd.DataFrame({
    'Name':['A','B','C','D'],
    'DSA':[40,50,60,70],
    'OOP':[80,90,100,60],
    'DS':[80,80,80,80],
})

print(df)

# melt basically used to represent dataframe in a different way sometimes pipelines need this

df1=df.melt(id_vars="Name",value_name="Subject",value_vars=["DSA","OOP","DS"],var_name="Score")
print(df1)