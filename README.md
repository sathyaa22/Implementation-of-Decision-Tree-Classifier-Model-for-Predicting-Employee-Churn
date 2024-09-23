# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the program.

Step 2: Import pandas module and import the required data set.

Step 3: Find the null values and count them.

Step 4: Count number of left values.

Step 5: From sklearn import LabelEncoder to convert string values to numerical values.

Step 6: From sklearn.model_selection import train_test_split.

Step 7: Assign the train dataset and test dataset.

Step 8: From sklearn.tree import DecisionTreeClassifier.

Step 1: Use criteria as entropy.

Step 9: From sklearn import metrics.

Step 10: Find the accuracy of our model and predict the require values.

Step 11: Stop the program.

## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SATHYAA R
RegisterNumber: 212223100052

```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```


## Output:

![Screenshot 2024-09-23 110329](https://github.com/user-attachments/assets/78cfdc70-0cb8-4327-86c3-56c7db774f01)

![Screenshot 2024-09-23 110342](https://github.com/user-attachments/assets/88633f49-eb80-4d49-b7c8-40c5b0e79365)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
