# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np


train_images=[[1,1,1],[1,1,1],[1,1,1]],[[0,1,0],[1,0,1],[0,1,0]],[[1,0,1],[0,1,0],[1,0,1]]
train_images=np.array(train_images)
train_labels=[.9,0.9,0.0]
train_labels=np.array(train_labels)
print(train_images.shape)
print(train_images[0])
print(train_labels[0])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(3, 3)),
    keras.layers.Dense(9, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=9)

predictions = model.predict(train_images)

print('pridiction:',predictions)
temp=[[[0,1,0],[1,1,1],[0,1,0]]]
temp=np.array(temp)
predictions = model.predict(temp)

print('pridiction:',predictions)