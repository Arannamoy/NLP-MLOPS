import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


data=sns.load_dataset("iris")
df=pd.DataFrame(data)
x_train=df.iloc[:,:-1]
y_train=df.iloc[:,-1]

model=RandomForestClassifier()
model.fit(x_train,y_train)

prediction=model.predict(x_train)
score=model.score(x_train.tail(10),y_train.tail(10))

print(score)