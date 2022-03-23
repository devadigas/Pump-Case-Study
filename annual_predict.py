
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Quality_Control.csv')

X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
regressor = RandomForestRegressor(n_estimators=50, random_state=0)
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
#Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))
print(Y_pred)
print(regressor.score(X_test,Y_test))
