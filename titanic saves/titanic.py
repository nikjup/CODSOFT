import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv("C:\\Users\\abc\\Downloads\\titanic.csv")
print(data)
print(data.shape)
print(data.info())
print(data.isnull().sum())
data=data.drop(columns='Cabin',axis=1)
print(data)
print(data['Age'].fillna(data['Age'].mean(),inplace=True))
print(data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True))
print(data['Fare'].fillna(data['Fare'].mode()[0],inplace=True))
print(data.isnull().sum().sum())
print(data['Survived'].value_counts())
print(data.describe())
sns.set()
sns.countplot(x='Survived',data=data)
sns.countplot(x='Sex',data=data)
sns.countplot(x='Sex',hue='Survived',data=data)
sns.countplot(x='Pclass',data=data)
sns.countplot(x='Pclass',hue='Survived',data=data)
print(data['Sex'].value_counts())
print(data['Embarked'].value_counts())
print(data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True))
print(data)
X=data.drop(columns=['PassengerId','Name','Ticket'],axis=1)
Y=data['Survived']
print(X)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
model=LogisticRegression()
print(model.fit(X_train,Y_train))
X_train_prediction=model.predict(X_train)
print(X_train_prediction)
train_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print("Accuracy Score of training data: ",train_data_accuracy)
X_test_prediction=model.predict(X_test)
print(X_test_prediction)
test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print("Accuracy score of testing data:",test_data_accuracy)
