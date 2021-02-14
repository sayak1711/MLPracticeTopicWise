import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

dataset_path = 'D://Work//MLCodes//Machine Learning A-Z (Codes and Datasets)//Part 2 - Regression//Section 5 - Multiple Linear Regression//Python/50_Startups.csv'

dataset = pd.read_csv(dataset_path)
X = dataset.iloc[:, :-1].values  # to make X into a 2D array we have to do like this instead of just column number
Y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

# split data to train, test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# fit training data on regressor
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(y_pred.shape)
print(Y_test.shape)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))  # see them adjacent to each other