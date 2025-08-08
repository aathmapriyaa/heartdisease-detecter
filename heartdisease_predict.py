import pandas as pd 
import sklearn.neighbors as knn 
mydata = pd.read_csv("heart.csv") 
x=mydata[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
y=mydata[["target"]] 
model=knn.KNeighborsClassifier(n_neighbors=5) 
model.fit(x,y)
getAge=int(input("Enter the age of the patient: "))
getSex=int(input("Enter sex: ")) 
getcp=int(input("Enter cp: ")) 
getTrestbps=int(input("Enter Trestbps: "))
getChol=int(input("Enter serum chol: "))
getFbs=int(input("Enter Fbs: "))
getRestecg=int(input("Enter Restecg: "))
getThalach=int(input("Enter Thalach: "))
getExang=int(input("Enter Exang: "))
getOldpeak=float(input("Enter Oldpeak: "))
getSlope=int(input("Enter slope: "))
getCa=int(input("Enter Ca: "))
getThal=int(input("Enter thal ")) 
 
target=model.predict([[getAge, getSex, getcp, getTrestbps, getChol, getFbs, getRestecg, getThalach, getExang, getOldpeak, getSlope, getCa, getThal]])
if (target[0] == 1):
    print("The patient is likely to have heart disease.")
else:
    print("The patient is likely to be healthy.") 