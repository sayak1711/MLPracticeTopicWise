import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

dataset_path = 'D://Work//MLCodes//Machine Learning A-Z (Codes and Datasets)//Part 3 - Classification//Section 14 - Logistic Regression//Python//Social_Network_Ads.csv'

dataset = pd.read_csv(dataset_path)
X = dataset.iloc[:, :-1].values  # to make X into a 2D array we have to do like this instead of just column number
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(Y_test, y_pred))
print(accuracy_score(Y_test, y_pred))
