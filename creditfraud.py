import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
file=pd.read_csv("C:\\Users\\abc\\Downloads\\creditcard.csv")
print(file.head(10))
print(file.describe())
print(file.isnull().sum())
print(file['Class'].value_counts())
normal=file[file.Class==0]
fraud=file[file.Class==1]
print(normal.shape)
print(normal.Amount.describe())
print(fraud.Amount.describe())
print(file.groupby('Class').mean())
normal_sample=normal.sample(n=492)
new_file=pd.concat([normal_sample,fraud],axis=0)
print(new_file.head(10))
print(new_file['Class'].value_counts())
print(new_file.groupby('Class').mean())
X=new_file.drop(columns='Class',axis=1)
Y=new_file['Class']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
model=LogisticRegression()
print(model.fit(X_train,Y_train))
X_train_prediction=model.predict(X_train)
training_data_acuracy=accuracy_score(X_train_prediction,Y_train)*100
print(training_data_acuracy)
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)*100
print(test_data_accuracy)




