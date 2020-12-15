#!/usr/bin/env python
# coding: utf-8

# # Keras Multivariable Regression NN
# #Reference: https://datascienceplus.com/keras-regression-based-neural-networks/



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor



#Get dataset, print shape and view
data = pd.read_csv("./vehicle_collision_data.csv")
print(data.shape)
data.head()



# preprocess by removing any NaN values
data = data.dropna()
print(data.shape)
data.head()



#Drop non needed columns and Y value for INPUT
x = data.drop(['Unnamed: 0','COLLISION',"HOST_LAT","HOST_LON","GUEST_LAT","GUEST_LON","HOST_DIRECTION","HOST_SPEED","INTERSECTION_LAT","INTERSECTION_LON"], axis = 1).to_numpy()
print(x.shape)
print(x)



# keep Collision true/false as output 
y = data['COLLISION'].to_numpy()
y = np.reshape(y,(-1,1))
print(y.shape)
print(y)



#normalize the input and output!
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()



# transform the data
print(scaler_x.fit(x))
xscale=scaler_x.transform(x)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)



#train the data, 
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)



#create NN with 2 hidden layers, one with 6 nodes and one with 4 nodes
model = Sequential()
model.add(Dense(6, input_dim=7, kernel_initializer='normal', activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()



#compiles the Training 
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])



#test the training and fit the model
history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)



#format and show key values
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



# random point for prediction
Xnew = np.array([[0.100278,7.220009,326,80,15,0.058550,7.22]])
Xnew= scaler_x.transform(Xnew)
#predict!
ynew= model.predict(Xnew)
#invert normalize
ynew = scaler_y.inverse_transform(ynew) 
Xnew = scaler_x.inverse_transform(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))





