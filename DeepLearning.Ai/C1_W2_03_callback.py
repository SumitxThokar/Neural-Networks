# Importing necessary modules
import tensorflow as tf
from tensorflow import keras
# Importing fashion_MNIST data
fmnist=tf.keras.datasets.fashion_mnist
# Load the data
(x_train,y_train),(x_test,y_test)=fmnist.load_data()
# Normalize the pixel values
x_train,x_test=x_train/255.0,x_test/255.0
# Creating callback class
class mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('loss')<0.3):
            print('Loss is lower than 0.3 so cancelling training!')
            self.model.stop_training=True
callbacks=mycallbacks()
# Define the model.
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax),
])
# Compile the model.
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#Train the model
model.fit(x_train,y_train,epochs=10,callbacks=[callbacks])




