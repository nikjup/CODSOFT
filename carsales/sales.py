import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data_file=pd.read_csv("C:\\Users\\abc\\Downloads\\sales.csv")
print(data_file.head(15))
print(data_file.shape)
print(data_file.describe())
print(data_file.isnull().sum())
print(sns.set())
data_file['TV'].hist()
data_file['Newspaper'].hist()
data_file['Radio'].hist()
sns.pairplot(data_file,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')
sns.heatmap(data_file.corr(),annot=True,cmap='coolwarm')
X=data_file.drop(columns='Sales')
Y=data_file['Sales']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=3)
model=LinearRegression()
model.fit(X_train,Y_train)
prediction=model.predict(X_test)
print(prediction)
model.intercept_
model.coef_
accuracy_score=model.score(X_test,Y_test)*100
print(accuracy_score)
