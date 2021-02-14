import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset_path = 'D://Work//MLCodes//Machine Learning A-Z (Codes and Datasets)//Part 2 - Regression//Section 6 - Polynomial Regression//Python//Position_Salaries.csv'

dataset = pd.read_csv(dataset_path)
X = dataset.iloc[:, 1:-1].values  # to make X into a 2D array we have to do like this instead of just column number
Y = dataset.iloc[:, -1].values
Y = Y.reshape(-1, 1)

# here we have to apply feature scaling unlike polynomial regression because there are no coefficients with the features
# which will compensate for the difference in scale

sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

svregressor = SVR()
svregressor.fit(X, Y)

plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(svregressor.predict(X)), color='blue')
plt.show()
