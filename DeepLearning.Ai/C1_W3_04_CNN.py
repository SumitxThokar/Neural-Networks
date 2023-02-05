import tensorflow as tf                                                        # Importing necessary libraires
from tensorflow import keras
fmist=tf.keras.datasets.fashion_mnist                                          # Load the data
(training_images,training_labels),(test_images,test_labels)=fmist.load_data()
training_images=training_images/255.0                                          # Normalize the pixel
test_images=test_images/255.0
class mycallback(tf.keras.callbacks.Callback):                                 # Callback function
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('loss')<0.22):
            print('\nCancelling further training as loss is lower than 0.22\n')
            self.model.stop_training=True
callbacks=mycallback()                                                         # Callback instance
model=tf.keras.Sequential([                                                    # Define the model
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),  # CNN and pooling layers
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax'),
])
model.summary()                                                                             # Summarize the model.
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) # Compile the model
model.fit(training_images,training_labels,epochs=6,callbacks=[callbacks])                   # Train the model
model.evaluate(test_images,test_labels)                                                     # Evaluate the model's performance


