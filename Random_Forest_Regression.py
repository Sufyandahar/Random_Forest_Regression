# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read csv files
dataset = pd.read_csv('Position_Salaries.csv')
# print(dataset)

# slicing dataset

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
# print(X)
# print(y)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

################################  Prediction ##################################
predction = regressor.predict([[10]])
print("Predicted value :", predction)

# visualizing the regression result ( for high resolution and smoother result)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color= 'blue')
plt.title("Truth or Bluff (Random Forest Regression")
plt.xlabel('Position level')
plt.ylabel("Salary")
plt.show()