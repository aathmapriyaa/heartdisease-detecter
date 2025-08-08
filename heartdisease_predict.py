import pandas as pd 
import sklearn.neighbors as knn 
mydata = pd.read_csv("heart.csv") 
x=mydata[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
y=mydata[["target"]] 
model=knn.KNeighborsClassifier(n_neighbors=5) 
model.fit(x,y)
print(model.predict([[52,1,0,125,212,0,1,168,0,1,2,2,3]])) 
target=model.predict([[52,1,0,125,212,0,1,168,0,1,2,2,3]])
if target[0] == 1:
    print("The patient is likely to have heart disease.")
else:
    print("The patient is likely to be healthy.")