#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:00:03 2025

@author: freya
利用残差来做
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


# In[data preprocess]
dataset = pd.read_csv("/Users/freya/Study/self-learning/kaggle_digit-recognizer/train.csv")
X = dataset.drop('label', axis=1).values
y_real = dataset['label']

# standard value   
X = X/255
X = X.reshape(-1,28,28,1) # reshape to 28*28
y_real = to_categorical(y_real, num_classes = 10) # one-hot，convert to [0,0,1,...]

del dataset


# In[CNN model]
num_outputs = 10
num_hidden1 = 25

initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
model = tf.keras.models.Sequential([
    # convolution layer
    tf.keras.layers.Conv2D(20, kernel_size=7, padding='same', 
                           kernel_initializer=initializer,activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=3),
    tf.keras.layers.Conv2D(10, kernel_size=4, padding='same', 
                           kernel_initializer=initializer,activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=3),
    # linear
    tf.keras.layers.Flatten(), # input layer
    tf.keras.layers.Dense(
        num_hidden1, activation='relu', 
        kernel_initializer=initializer
        ), # hiden layer
    tf.keras.layers.Dense(num_outputs)]
    ) # output layer

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-3),
    metrics=['accuracy']
)

callback = tf.keras.callbacks.EarlyStopping(patience=10, mode='min', 
                                            restore_best_weights=True, start_from_epoch=50)

history = model.fit(X, y_real, batch_size=30, epochs=100, validation_split=0.1, callbacks=[callback])


# test session
testset = pd.read_csv("/Users/freya/Study/self-learning/kaggle_digit-recognizer/test.csv")
testset = testset/255
testset = testset.values.reshape(-1,28,28,1) # reshape 到28*28

y_pred = model.predict(testset)
y_pred_digits = np.argmax(y_pred, axis=1)


sample_submission_df = pd.read_csv('/Users/freya/Study/self-learning/kaggle_digit-recognizer/sample_submission.csv')
sample_submission_df['Label'] = y_pred_digits
sample_submission_df.to_csv('/Users/freya/Study/self-learning/kaggle_digit-recognizer/submission.csv', index=False)
