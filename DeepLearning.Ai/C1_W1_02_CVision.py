#Importing necessary modules
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
fmnist=tf.keras.datasets.fashion_mnist                                          #Loading the Fashion MNIST dataset
(training_images,training_labels),(test_images,test_labels)=fmnist.load_data()  #Load the training and test split of the fashion MNIST dataset
index=20000                                                                     #index equivalent to the image pixel array to print in that index
np.set_printoptions(linewidth=320)                                              #sets the number of characters per rows when printing    
print(f'LABEL:{training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n{training_images[index]}')
plt.imshow(training_images[index])                                              #Visualize the image
training_images=training_images/255.0                                           #Normalizing the values as model best work in value 0 to 1.
test_images=test_images/255.0
model=tf.keras.Sequential([                                                     #Building model with 3 layers.
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),                                  #Compiling the model.
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(training_images,training_labels,epochs=5)                            #Training the training data.
model.evaluate(test_images,test_labels)                                        # Evaluate the model on unseen data.






