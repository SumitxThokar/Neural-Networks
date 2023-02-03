
# Importing Necessary Libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras
# To check the the tensorflow version.
print(tf.__version__)            
# Creating the simplest possible neural network. It has 1 layer with 1 neuron, and the input shape to it is just 1 value.
model= tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
# Compile the model.
model.compile(optimizer='sgd',loss='mean_squared_error')
# Labelled data.
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
# Train the model with data.
model.fit(xs,ys,epochs=600)
# Predict the output for new value
print(model.predict([10.0]))
# Since Neural networks deal with probabilites, the output of 10.0 is not exactly 19. Given the data that we fed the model,
# with it calculated that there is a very high probability that the realtion between x and y is y=2x-1 but with only 6 data
# points we can't know for sure. As a result for 10 is very close to 19 but not exactly 19.
