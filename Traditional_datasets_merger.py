from __future__ import absolute_import, division, print_function
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.models import Model

import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd


X_train= pd.read_csv('./15.csv')



values = X_train.values

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning

train = scaled[:, :]
# split into input and outputs
X, y = train[:, :-1], train[:, -1]

test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)

features = fit.transform(X)
# Summarize selected features
print(features[0:5,:])


model = Sequential()
model.add(Dense(100, input_dim=16, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
from keras.optimizers import SGD
opt = SGD(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X, y, epochs=200, batch_size=100)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


