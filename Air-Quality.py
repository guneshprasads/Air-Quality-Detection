#Importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("datasets_153_328_AirQualityUCI.csv")

#Encoding the data and time object 
dataset['Time'] = pd.to_datetime(dataset['Time']).dt.hour
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset.insert(1, "year", dataset.Date.dt.year, True)
dataset.insert(2, "month", dataset.Date.dt.month, True)
dataset.insert(3, "Day", dataset.Date.dt.day, True)

#Droping the unwanted coloumn
dataset.drop('Date', axis=1, inplace=True)

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#Missing Value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=-200,strategy='mean')
X[:,4:] = imputer.fit_transform(X[:,4:])
print(X)

#Checking the Null Values
dataset.isnull().any()

#Categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le=LabelEncoder()
Y=le.fit_transform(Y)
print(Y)
Y=Y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
Y=ohe.fit_transform(Y)

#spliting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 4:] = sc.fit_transform(X_train[:, 4:])
X_test[:, 4:] = sc.transform(X_test[:, 4:])
print(X_train)
print(X_test)


#Training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predecting the Model
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

#checking the R-Square Value
from sklearn.metrics import r2_score
r2_score(Y_test, Y_pred)
