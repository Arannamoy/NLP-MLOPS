import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("./6_Scikit_Learn/Data/housing.csv")
df=pd.DataFrame(data)

x=df.iloc[:,:-1]
y=df.iloc[:,-1]
model=RandomForestClassifier()
model.fit(x,y)

prediction=model.predict([[-122.23 ,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,452600.0]])
print(prediction)