import numpy as np
import sklearn.datasets

breast_cancer = sklearn.datasets.load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target
print(X.shape)
print(Y.shape)

#print(X.mean())
#print(Y.mean())

import pandas as pd

data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
data['class'] = breast_cancer.target
print(data.head())
print(breast_cancer.target_names)
data.groupby('class').mean()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
# print(X.shape, X_train.shape, X_test.shape)
# print(Y.shape, Y_train.shape, Y_test.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, stratify = Y, random_state = 1)
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)

print(X.mean(), X_train.mean(), X_test.mean())
print(X_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score

test_pred = model.predict(X_train)
acc_test = accuracy_score(Y_train, test_pred)
print('accuracy_score of test Data:', acc_test)

input_data = ()

inp_as_np_array = np.asarray(input_data)
print(input_data)

reshape_arr = inp_as_np_array.reshape(1, -1)

## PREDICTION
pred = model.predict(reshape_arr)
print(pred)

if (pred[0] == 0):
    print("The Cancer is Melignant")
else:
    print("The Cancer is Benign")
