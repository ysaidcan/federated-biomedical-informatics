from __future__ import absolute_import, division, print_function
from sklearn.preprocessing import MinMaxScaler
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


X_train= pd.read_csv('./daily_frames_HR.csv')
values = X_train.values
values = values.astype('float32')
train = values[:, :]
# split into input and outputs
X, y = train[:, :-1], train[:, -1]


def create_compiled_keras_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(
      2, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(16,)), tf.keras.layers.Dense(100, activation='relu'), tf.keras.layers.Dense(100, activation='relu'),  tf.keras.layers.Dense(1, activation='sigmoid')])

  return model

def model_fn():
  keras_model = create_compiled_keras_model()

  keras_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['BinaryAccuracy'])

  X_train = pd.read_csv('./daily_frames_HR.csv')
  values = X_train.values
  values = values.astype('float32')
  train = values[:, :]
  # split into input and outputs
  X, y = train[:, :-1], train[:, -1]


  X = pd.DataFrame(X)
  y = pd.DataFrame(y)

  sample_batch = collections.OrderedDict([('x', X), ('y', y)])
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)


iterative_process = tff.learning.build_federated_averaging_process(model_fn)
print(str(iterative_process.initialize.type_signature))
state = iterative_process.initialize()

X2_train= pd.read_csv('./lab_frames_HR.csv')
values2 = X2_train.values
values2 = values2.astype('float32')
train2 = values2[:, :]
# split into input and outputs
X2, y2 = train2[:, :-1], train2[:, -1]



X3_train= pd.read_csv('./ILKYAR.csv')
values3 = X3_train.values
values3 = values3.astype('float32')
train3 = values3[:, :]
# split into input and outputs
X3, y3 = train3[:, :-1], train3[:, -1]

X4_train= pd.read_csv('./affectech.csv')
values4 = X4_train.values
values4 = values4.astype('float32')
train4 = values4[:, :]
# split into input and outputs
X4, y4 = train4[:, :-1], train4[:, -1]


X4=pd.DataFrame(X4)
y4=pd.DataFrame(y4)


X3=pd.DataFrame(X3)
y3=pd.DataFrame(y3)

X2=pd.DataFrame(X2)
y2=pd.DataFrame(y2)

X=pd.DataFrame(X)
y=pd.DataFrame(y)


dataset = tf.data.Dataset.from_tensor_slices((X2.values, y2.values))

dataset2= tf.data.Dataset.from_tensor_slices((X.values, y.values))

dataset3= tf.data.Dataset.from_tensor_slices((X3.values, y3.values))

dataset4= tf.data.Dataset.from_tensor_slices((X4.values, y4.values))


list = [ dataset2.batch(1), dataset3.batch(1)]



print(str(iterative_process.initialize.type_signature))


state, metrics = iterative_process.next(state, list)
print('round  1, metrics={}'.format(metrics))

for round_num in range(2, 111):
  state, metrics = iterative_process.next(state, list)
  print('round {:2d}, metrics={}'.format(round_num, metrics))




