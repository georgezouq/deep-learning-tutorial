from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

'''
Input shape
Model needs to know what input shape it should expect. The first layer in a `Sequential` model
(and only the first) needs to receive information about its input shape.

- Pass an `input_shape` argument to the first layer
- For 2D layers(as `Dense`), support specification of their input shape via the argument `input_dim`
and for 3D support both input_dim and input_length.
'''

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(6, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

result_num = 33

'''
Compilation
'''

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

'''
Training Set
'''

x_train = np.random.randint(33, size=(1000, 100), replace=False)
y_train = np.random.randint(2, size=(1000, result_num))

x_test = np.random.random((1000, 100))
y_test = np.random.randint(2, size=(1000, result_num))

'''
Training

model.fit(x, y, epochs, batch_size)
'''

model.fit(x_train, y_train, epochs=10, batch_size=33)

'''
Get Score
'''

score = model.evaluate(x_test, y_test, batch_size=33)
print('score:', score)

'''
For Predict
'''

test = np.random.random((1, 100))
print('predict result:', model.predict(test))