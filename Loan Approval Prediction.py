import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset =  pd.read_csv("D:/Data Analytics Projects/loan-train.csv")
#print(dataset.head())

# print(dataset.shape)
# print(dataset.info())
# print(dataset.describe())
# print(pd.crosstab(dataset['Credit_History'],dataset['Loan_Status'],margins=True))

#dataset.boxplot(column='ApplicantIncome')
#dataset['ApplicantIncome'].hist(bins=20) #Right skewed
#dataset['CoapplicantIncome'].hist(bins=20) #Right skewed
#dataset.boxplot(column='ApplicantIncome',by = 'Education')

#dataset.boxplot(column='LoanAmount')
#dataset['LoanAmount'].hist(bins=20)

# Normalizing using log function
dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
#dataset['LoanAmount_log'].hist(bins=20)

## Handling null values
#print(dataset.isnull().sum())
#For catagorical variables mode is used to fill the values
# Remember to apply index on the mode
dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)

# Replacing by mean values for loan amount
dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())
#print(dataset.isnull().sum())

#Normalizing total income
# Total income = sum of applicant income and coapplicant income
dataset['TotalIncome']=dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])
dataset['TotalIncome_log'].hist(bins=20)
#print(dataset.head())

X = dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y = dataset.iloc[:,12].values

# Splitting training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#print(X_train)

# Using label encoder
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
for i in range(0,5):
    X_train[:,i]= labelencoder_X.fit_transform(X_train[:,i])
X_train[:,7]= labelencoder_X.fit_transform(X_train[:,7])

labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)

for i in range(0,5):
    X_test[:,i]= labelencoder_X.fit_transform(X_test[:,i])
X_test[:,7]= labelencoder_X.fit_transform(X_test[:,7])

y_test = labelencoder_y.fit_transform(y_test)

# Scaling the data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

#Applying the algorithms
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
DTClassifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
DTClassifier.fit(X_train,y_train)
y_pred = DTClassifier.predict(X_test)

from sklearn import metrics
print('The accuracy of decision tree is: ',metrics.accuracy_score(y_pred,y_test))


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(X_train,y_train)
y_pred = NBClassifier.predict(X_test)
print('The accuracy of NB is: ',metrics.accuracy_score(y_pred,y_test))

#Import the test data
test_data = pd.read_csv("D:/Data Analytics Projects/loan-test.csv")
# handling missing values
test_data['Gender'].fillna(test_data['Gender'].mode()[0],inplace=True)
test_data['Married'].fillna(test_data['Married'].mode()[0],inplace=True)
test_data['Dependents'].fillna(test_data['Dependents'].mode()[0],inplace=True)
test_data['Self_Employed'].fillna(test_data['Self_Employed'].mode()[0],inplace=True)
test_data['Loan_Amount_Term'].fillna(test_data['Loan_Amount_Term'].mode()[0],inplace=True)
test_data['Credit_History'].fillna(test_data['Credit_History'].mode()[0],inplace=True)
test_data['LoanAmount_log']=np.log(test_data['LoanAmount'])
test_data.LoanAmount = test_data.LoanAmount.fillna(test_data.LoanAmount.mean())
test_data.LoanAmount_log = test_data.LoanAmount_log.fillna(test_data.LoanAmount_log.mean())
#print(test_data.isnull().sum())
test_data['TotalIncome']=test_data['ApplicantIncome'] + test_data['CoapplicantIncome']
test_data['TotalIncome_log']=np.log(test_data['TotalIncome'])

test = test_data.iloc[:,np.r_[1:5,9:11,13:15]].values
for i in range(0,5):
    test[:,i]= labelencoder_X.fit_transform(test[:,i])
test[:,7]= labelencoder_X.fit_transform(test[:,7])
#print(test)

test=ss.fit_transform(test)
pred = NBClassifier.predict(test)

print(pred)