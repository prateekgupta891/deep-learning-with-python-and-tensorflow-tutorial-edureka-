import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston

boston = load_boston()
bos = pd.DataFrame(boston.data)

print(bos.head(5))
bos.columns = boston.feature_names
bos['Price'] = boston.target
Y = bos['Price']
X = bos.drop(['Price'],axis = 1)
print(X.head(5))

print(Y.head(5))

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state = 5)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

lr = LinearRegression()
a = lr.fit(X_train,Y_train)
Y_train_pred = lr.predict(X_train)
Y_test_pred = lr.predict(X_test)
df = pd.DataFrame(Y_test_pred, Y_test)
print(df)
from sklearn.metrics import mean_squared_error as mse
print(mse(Y_test,Y_test_pred))

fig, ax = plt.subplots()
ax.scatter(Y_test,Y_test_pred)
ax.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()],'k--',lw =3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
