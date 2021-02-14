import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset_path = 'D://Work//MLCodes//Machine Learning A-Z (Codes and Datasets)//Part 2 - Regression//Section 4 - Simple Linear Regression//Python/Salary_Data.csv'

dataset = pd.read_csv(dataset_path)
X = dataset.iloc[:, :-1].values  # to make X into a 2D array we have to do like this instead of just column number
Y = dataset.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
#print(X_train, Y_train)
#print(X_test, Y_test)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
print(y_pred)
print(Y_test)

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.show()
