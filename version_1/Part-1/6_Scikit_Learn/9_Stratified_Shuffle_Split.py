# Suppose our dataset have two inland in last 2 row. If we trained 60% data from datasets these 2 can be untrained thats why we trained one inland  data for training and one for testing

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

data=pd.read_csv("./6_Scikit_Learn/Data/housing.csv")

df=pd.DataFrame(data=data)

df['income_cat']=pd.cut(df['median_income'],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(df,df['income_cat']):
    strat_train_set=df.loc[train_index]
    strat_test_set=df.loc[test_index]

strat_test_set["income_cat"].value_counts().sort_index().plot.bar(rot=0,grid=True)
plt.title("Strat Test Set")
plt.xlabel("Income Category")
plt.ylabel("Number Of Instances")

plt.savefig("./6_Scikit_Learn/Data/9_Stratified_Shuffle.png")