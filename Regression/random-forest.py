import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

dataset_path = 'D://Work//MLCodes//Machine Learning A-Z (Codes and Datasets)//Part 2 - Regression//Section 6 - Polynomial Regression//Python//Position_Salaries.csv'

dataset = pd.read_csv(dataset_path)
X = dataset.iloc[:, 1:-1].values  # to make X into a 2D array we have to do like this instead of just column number
Y = dataset.iloc[:, -1].values

rfregressor = RandomForestRegressor(n_estimators=10, random_state=0)
rfregressor.fit(X, Y)

plt.scatter(X, Y, color='red')
plt.plot(X, rfregressor.predict(X), color='blue')

plt.show()
