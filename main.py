import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import tensorflow as tf
import matplotlib.pyplot as plt

# loading data

data = pd.read_csv('data.csv')

# checking missing values

print(data.isnull().sum())
# no missing values in each column

# preprocessing data
# declaring X our independent variables
X = data.drop('Age', axis=1)

# remove categorize column 'Id'
# X = X.drop('Sex', axis=1)
X = X.drop('id', axis=1)
# change Sex column to other 3 columns with values of 0 and 1
X = pd.get_dummies(X, columns=['Sex'], dtype=float)
print(X.head())

# Standardize the features
names = X.columns
scaled_columns = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']
scaler = StandardScaler()
X[scaled_columns] = scaler.fit_transform(X[scaled_columns])
print(X.head())
# declaring y our target variable
y = data['Age']

print(y.head())

# splitting date for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# building Random forest regression model

reg = RandomForestRegressor(random_state=0)
reg.fit(X_train, y_train)

# tensorflow model

tensor = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1)
])

tensor.compile(optimizer='Adam', loss='mean_squared_error')
tensor.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Linear Regression


linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)

# Results
print('\nRandom forest regression model')
# mean square error
prediction = reg.predict(X_test)
print("Mean squared error: %.3f" % mean_squared_error(y_test, prediction))

# mean absolute error
mae = mean_absolute_error(y_test, prediction)
print('Mean Absolute Error:', mae)

# R2 score
r2 = r2_score(y_test, prediction)
print('R2 score: ', r2)

print('\nTensorflow Sequential model')
# mean square error

prediction2 = tensor.predict(X_test)
print("Mean squared error: %.3f" % mean_squared_error(y_test, prediction2))

# mean absolute error
mae = mean_absolute_error(y_test, prediction2)
print('Mean Absolute Error:', mae)

# R2 score
r2 = r2_score(y_test, prediction2)
print('R2 score: ', r2)

print('\nLinear Regression model')
# mean square error

prediction3 = linear.predict(X_test)
print("Mean squared error: %.3f" % mean_squared_error(y_test, prediction3))

# mean absolute error
mae = mean_absolute_error(y_test, prediction3)
print('Mean Absolute Error:', mae)

# R2 score
r2 = r2_score(y_test, prediction3)
print('R2 score: ', r2)

# graphical representation

data.plot(kind='scatter', x='Weight', y='Age', figsize=(15, 5), xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
plt.show(block=True)

data.plot(kind='scatter', x='Length', y='Age', figsize=(15, 5), xticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3 , 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2])
plt.show(block=True)
