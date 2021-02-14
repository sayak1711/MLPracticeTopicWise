import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

dataset = pd.read_csv('D://Work//MLCodes//Machine Learning A-Z (Codes and Datasets)//Part 5 - Association Rule '
                      'Learning//Section 28 - Apriori//Python//Market_Basket_Optimisation.csv', header=None)
transactions = []  # each row is list of items bought together
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2,
                max_length=2)

results = list(rules)


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


resultsinDataFrame = pd.DataFrame(inspect(results),
                                  columns=['Product 1', 'Product 2', 'Support'])

print(resultsinDataFrame.nlargest(n=10, columns='Support'))
