# IMPORTING LIBRARIES
import panda as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('california_housing_test')
df.head()

y = df['total_rooms']
x = df [['longitude', 'latitude', 'housing_median_age']]

def CasesReg(x,y):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(x,y)
    return reg

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

reg = CasesReg(x_train, y_train)


y_pred = reg.predict(x_test)
y_pred

reg.score(x_test, y_test)


plt.plot(x, reg.predict(x), '*')
plt.legend(labels=['longitude', 'latitude', 'housing_median_age'])
plt.xlabel("Long Rate")
plt.ylabel("total_rooms")
plt.show()
