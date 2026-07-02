from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

data=pd.read_csv("./6_Scikit_Learn/Data/housing.csv")

df=pd.DataFrame(data=data)

df["income_cat"]=pd.cut(df["median_income"],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

split=StratifiedShuffleSplit()

for train_index,test_index in split.split(df,df["income_cat"]):
    strat_train_set=df.iloc[train_index]
    strat_test_set=df.iloc[test_index]

# Lets remove income_cat column 

for st in (strat_train_set,strat_test_set):
    st.drop("income_cat",axis=1,inplace=True)


df1=strat_train_set.copy()
ax=df1.plot(kind="scatter",x="latitude",y="longitude",grid=True,alpha=0.2)
df1.drop(labels="ocean_proximity",inplace=True,axis=1)

attributes=["housing_median_age","median_income","median_house_value"]
scatter_matrix(df[attributes],figsize=(12,8))