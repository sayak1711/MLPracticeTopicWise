import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

dataset_path = 'D://Work//MLCodes//Machine Learning A-Z (Codes and Datasets)//Part 2 - Regression//Section 6 - Polynomial Regression//Python//Position_Salaries.csv'

dataset = pd.read_csv(dataset_path)
X = dataset.iloc[:, 1:-1].values  # to make X into a 2D array we have to do like this instead of just column number
Y = dataset.iloc[:, -1].values

# training linear regression on whole dataset
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# train polynomial regression on whole dataset
poly_reg = PolynomialFeatures(degree=4)  # generating more features using higher powers of x till 4
X_poly = poly_reg.fit_transform(X)
#print(X_poly)
#print(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.plot(X, lin_reg2.predict(X_poly), color='green')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
