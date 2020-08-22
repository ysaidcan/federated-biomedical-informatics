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
from keras.optimizers import SGD

X_train= pd.read_csv('./personal_HR/1.csv')
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
  opt = SGD(lr=0.000001)

  keras_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['BinaryAccuracy'])

  X_train = pd.read_csv('./personal_HR/1.csv')
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

X2_train= pd.read_csv('./personal_HR/2.csv')
values2 = X2_train.values
values2 = values2.astype('float32')
train2 = values2[:, :]
# split into input and outputs
X2, y2 = train2[:, :-1], train2[:, -1]



X3_train= pd.read_csv('./personal_HR/3.csv')
values3 = X3_train.values
values3 = values3.astype('float32')
train3 = values3[:, :]
# split into input and outputs
X3, y3 = train3[:, :-1], train3[:, -1]

X4_train= pd.read_csv('./personal_HR/4.csv')
values4 = X4_train.values
values4 = values4.astype('float32')
train4 = values4[:, :]
# split into input and outputs
X4, y4 = train4[:, :-1], train4[:, -1]

X5_train= pd.read_csv('./personal_HR/5.csv')
values5 = X5_train.values
values5 = values5.astype('float32')
train5 = values5[:, :]
# split into input and outputs
X5, y5 = train5[:, :-1], train5[:, -1]


X6_train= pd.read_csv('./personal_HR/6.csv')
values6 = X6_train.values
values6 = values6.astype('float32')
train6 = values6[:, :]
# split into input and outputs
X6, y6 = train6[:, :-1], train6[:, -1]


X7_train= pd.read_csv('./personal_HR/7.csv')
values7 = X7_train.values
values7 = values7.astype('float32')
train7 = values7[:, :]
# split into input and outputs
X7, y7 = train7[:, :-1], train7[:, -1]


X8_train= pd.read_csv('./personal_HR/8.csv')
values8 = X8_train.values
values8 = values8.astype('float32')
train8 = values8[:, :]
# split into input and outputs
X8, y8 = train8[:, :-1], train8[:, -1]

X9_train= pd.read_csv('./personal_HR/9.csv')
values9 = X9_train.values
values9 = values9.astype('float32')
train9 = values9[:, :]
# split into input and outputs
X9, y9 = train9[:, :-1], train9[:, -1]

X10_train= pd.read_csv('./personal_HR/10.csv')
values10 = X10_train.values
values10 = values10.astype('float32')
train10 = values10[:, :]
# split into input and outputs
X10, y10 = train10[:, :-1], train10[:, -1]

X11_train= pd.read_csv('./personal_HR/11.csv')
values11 = X11_train.values
values11 = values11.astype('float32')
train11 = values11[:, :]
# split into input and outputs
X11, y11 = train11[:, :-1], train11[:, -1]

X12_train= pd.read_csv('./personal_HR/12.csv')
values12 = X12_train.values
values12 = values12.astype('float32')
train12 = values12[:, :]
# split into input and outputs
X12, y12 = train12[:, :-1], train12[:, -1]

X13_train= pd.read_csv('./personal_HR/13.csv')
values13 = X13_train.values
values13 = values13.astype('float32')
train13 = values13[:, :]
# split into input and outputs
X13, y13 = train13[:, :-1], train13[:, -1]

X14_train= pd.read_csv('./personal_HR/14.csv')
values14 = X14_train.values
values14 = values14.astype('float32')
train14 = values14[:, :]
# split into input and outputs
X14, y14 = train14[:, :-1], train14[:, -1]



X15_train= pd.read_csv('./personal_HR/15.csv')
values15 = X15_train.values
values15 = values15.astype('float32')
train15 = values15[:, :]
# split into input and outputs
X15, y15 = train15[:, :-1], train15[:, -1]

X16_train= pd.read_csv('./personal_HR/16.csv')
values16 = X16_train.values
values16 = values16.astype('float32')
train16 = values16[:, :]
# split into input and outputs
X16, y16 = train16[:, :-1], train16[:, -1]

X17_train= pd.read_csv('./personal_HR/17.csv')
values17 = X17_train.values
values17 = values17.astype('float32')
train17 = values17[:, :]
# split into input and outputs
X17, y17 = train17[:, :-1], train17[:, -1]


X18_train= pd.read_csv('./personal_HR/18.csv')
values18 = X18_train.values
values18 = values18.astype('float32')
train18 = values18[:, :]
# split into input and outputs
X18, y18 = train18[:, :-1], train18[:, -1]


X19_train= pd.read_csv('./personal_HR/19.csv')
values19 = X19_train.values
values19 = values19.astype('float32')
train19 = values19[:, :]
# split into input and outputs
X19, y19 = train19[:, :-1], train19[:, -1]


X20_train= pd.read_csv('./personal_HR/20.csv')
values20 = X20_train.values
values20 = values20.astype('float32')
train20 = values20[:, :]
# split into input and outputs
X20, y20 = train20[:, :-1], train20[:, -1]

X21_train= pd.read_csv('./personal_HR/21.csv')
values21 = X21_train.values
values21 = values21.astype('float32')
train21 = values21[:, :]
# split into input and outputs
X21, y21 = train21[:, :-1], train21[:, -1]

X22_train= pd.read_csv('./personal_HR/22.csv')
values22 = X22_train.values
values22 = values22.astype('float32')
train22 = values22[:, :]
# split into input and outputs
X22, y22 = train22[:, :-1], train22[:, -1]

X23_train= pd.read_csv('./personal_HR/23.csv')
values23 = X23_train.values
values23 = values23.astype('float32')
train23 = values23[:, :]
# split into input and outputs
X23, y23 = train23[:, :-1], train23[:, -1]

X24_train= pd.read_csv('./personal_HR/24.csv')
values24 = X24_train.values
values24 = values24.astype('float32')
train24 = values24[:, :]
# split into input and outputs
X24, y24 = train24[:, :-1], train24[:, -1]

X25_train= pd.read_csv('./personal_HR/25.csv')
values25 = X25_train.values
values25 = values25.astype('float32')
train25 = values25[:, :]
# split into input and outputs
X25, y25 = train25[:, :-1], train25[:, -1]


X26_train= pd.read_csv('./personal_HR/26.csv')
values26 = X26_train.values
values26 = values26.astype('float32')
train26 = values26[:, :]
# split into input and outputs
X26, y26 = train26[:, :-1], train26[:, -1]




X26=pd.DataFrame(X26)
y26=pd.DataFrame(y26)


X25=pd.DataFrame(X25)
y25=pd.DataFrame(y25)


X24=pd.DataFrame(X24)
y24=pd.DataFrame(y24)


X23=pd.DataFrame(X23)
y23=pd.DataFrame(y23)


X22=pd.DataFrame(X22)
y22=pd.DataFrame(y22)


X21=pd.DataFrame(X21)
y21=pd.DataFrame(y21)


X20=pd.DataFrame(X20)
y20=pd.DataFrame(y20)


X19=pd.DataFrame(X19)
y19=pd.DataFrame(y19)


X18=pd.DataFrame(X18)
y18=pd.DataFrame(y18)


X17=pd.DataFrame(X17)
y17=pd.DataFrame(y17)


X16=pd.DataFrame(X16)
y16=pd.DataFrame(y16)

X15=pd.DataFrame(X15)
y15=pd.DataFrame(y15)

X14=pd.DataFrame(X14)
y14=pd.DataFrame(y14)


X13=pd.DataFrame(X13)
y13=pd.DataFrame(y13)


X12=pd.DataFrame(X12)
y12=pd.DataFrame(y12)


X11=pd.DataFrame(X11)
y11=pd.DataFrame(y11)


X10=pd.DataFrame(X10)
y10=pd.DataFrame(y10)


X9=pd.DataFrame(X9)
y9=pd.DataFrame(y9)


X8=pd.DataFrame(X8)
y8=pd.DataFrame(y8)


X7=pd.DataFrame(X7)
y7=pd.DataFrame(y7)


X6=pd.DataFrame(X6)
y6=pd.DataFrame(y6)


X5=pd.DataFrame(X5)
y5=pd.DataFrame(y5)


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

dataset5= tf.data.Dataset.from_tensor_slices((X5.values, y5.values))

dataset6= tf.data.Dataset.from_tensor_slices((X6.values, y6.values))

dataset7= tf.data.Dataset.from_tensor_slices((X7.values, y7.values))

dataset8= tf.data.Dataset.from_tensor_slices((X8.values, y8.values))

dataset9= tf.data.Dataset.from_tensor_slices((X9.values, y9.values))

dataset10= tf.data.Dataset.from_tensor_slices((X10.values, y10.values))

dataset11= tf.data.Dataset.from_tensor_slices((X11.values, y11.values))

dataset12= tf.data.Dataset.from_tensor_slices((X12.values, y12.values))

dataset13= tf.data.Dataset.from_tensor_slices((X13.values, y13.values))

dataset14 = tf.data.Dataset.from_tensor_slices((X14.values, y14.values))

dataset15= tf.data.Dataset.from_tensor_slices((X15.values, y15.values))

dataset16= tf.data.Dataset.from_tensor_slices((X16.values, y16.values))

dataset17= tf.data.Dataset.from_tensor_slices((X17.values, y17.values))

dataset18= tf.data.Dataset.from_tensor_slices((X18.values, y18.values))

dataset19= tf.data.Dataset.from_tensor_slices((X19.values, y19.values))

dataset20= tf.data.Dataset.from_tensor_slices((X20.values, y20.values))

dataset21= tf.data.Dataset.from_tensor_slices((X21.values, y21.values))

dataset22= tf.data.Dataset.from_tensor_slices((X22.values, y22.values))

dataset23= tf.data.Dataset.from_tensor_slices((X23.values, y23.values))

dataset24= tf.data.Dataset.from_tensor_slices((X24.values, y24.values))

dataset25= tf.data.Dataset.from_tensor_slices((X25.values, y25.values))

dataset26= tf.data.Dataset.from_tensor_slices((X26.values, y26.values))


list = [ dataset.batch(1), dataset2.batch(1), dataset3.batch(1) , dataset4.batch(1), dataset5.batch(1), dataset6.batch(1), dataset7.batch(1),  dataset8.batch(1), dataset9.batch(1), dataset10.batch(1) , dataset11.batch(1), dataset12.batch(1), dataset13.batch(1) , dataset14.batch(1), dataset15.batch(1), dataset16.batch(1), dataset17.batch(1), dataset18.batch(1), dataset19.batch(1), dataset20.batch(1),  dataset21.batch(1), dataset22.batch(1), dataset23.batch(1), dataset24.batch(1), dataset25.batch(1), dataset26.batch(1)]



print(str(iterative_process.initialize.type_signature))


state, metrics = iterative_process.next(state, list)
print('round  1, metrics={}'.format(metrics))

for round_num in range(2, 111):
  state, metrics = iterative_process.next(state, list)
  print('round {:2d}, metrics={}'.format(round_num, metrics))




