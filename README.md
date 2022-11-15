# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard Libraries.<br>
2. Set variables for assigning dataset values.<br>
3. Import linear regression from sklearn.<br>
4. Assign the points for representing in the graph.<br>
5. Predict the regression for marks by using the representation of  the graph.<br>
6. Compare the graphs and hence we obtained the linear regression for the given datas.<br>

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M.Parshwanath
RegisterNumber:  212221230073
*/
import pandas as pd
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df)

X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)


```

# Output:
## Training dataset:
 
![Training dataset](https://user-images.githubusercontent.com/94165336/200157975-571e42e2-1cc5-4f7a-a4b1-1ee30c21ef18.png)

![Training dataset](https://user-images.githubusercontent.com/94165336/197789061-a95223c8-028d-42f6-927e-740fb46226aa.png)
## Test dataset:
![Test dataset](https://user-images.githubusercontent.com/94165336/197789111-a90533a0-3c4a-4e6c-b0f3-0c6d1a977e86.png)
![Test dataset](https://user-images.githubusercontent.com/94165336/197789217-3e9b33d2-6f07-44b8-aa74-c3b2c4897279.png) 
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
