from sklearn.tree import DecisionTreeClassifier


features=[["0.5","0"],["0.5","0"],["0.4","0"],["0.2","0"],["0.5","0"],["0.1","0"],["0.1","0"]]
labels=["Apple","Apple","Blueberry","Orange","Apple","Guava","Capcicum"]

model=DecisionTreeClassifier()
model.fit(features,labels)

prediction=model.predict([["0.5","2"],["0.3","0"],["1","1"],["0.1","0"]])
print(prediction)