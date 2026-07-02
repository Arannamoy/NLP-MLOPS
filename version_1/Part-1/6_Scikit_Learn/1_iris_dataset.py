import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data=sns.load_dataset("iris").copy()

df=pd.DataFrame(data)

x=df.iloc[:,:-1]
y=df.iloc[:,-1]


model=RandomForestClassifier()

model.fit(x,y) # for model trained

prediction=model.predict([[6.7,3.0,5.2,2.3]])

print(df.tail(5))
print(prediction)