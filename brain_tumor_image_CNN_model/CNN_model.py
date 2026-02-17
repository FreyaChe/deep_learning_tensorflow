#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:54:44 2026

@author: freya
"""


import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
import pandas as pd


# In[create dataset path]
# from pathlib import Path
# import random

# folders = ['glioma_tumor', 'meningioma_tumor', 'normal','pituitary_tumor']

# train_path = []
# val_path   = []
# test_path  = []

# for folder in folders:
#     path = Path('/Users/freya/Downloads/brain_tumor_image/4 classes/'+folder)
#     data_list = list(path.glob('*.jpg'))
#     random.shuffle(data_list)

#     n = len(data_list)
#     n_train = int(n * 0.8)
#     n_val   = int(n * 0.1)
#     n_test  = n - n_train - n_val  
    
#     train_path.extend(str(p.resolve()) for p in data_list[:n_train])
#     val_path.extend(str(p.resolve()) for p in data_list[n_train:n_train+n_val])
#     test_path.extend(str(p.resolve()) for p in data_list[n_train+n_val:])
    
    
# with open('/Users/freya/Downloads/brain_tumor_image/CNN_model/train_path.json', "w") as f:
#     json.dump(train_path, f)
# with open('/Users/freya/Downloads/brain_tumor_image/CNN_model/val_path.json', "w") as f:
#     json.dump(val_path, f)
# with open('/Users/freya/Downloads/brain_tumor_image/CNN_model/test_path.json', "w") as f:
#     json.dump(test_path, f)


# In[generate dataset]
with open('/Users/freya/Downloads/brain_tumor_image/CNN_model/train_path.json', "r") as f:
    train_path = json.load(f)
with open('/Users/freya/Downloads/brain_tumor_image/CNN_model/val_path.json', "r") as f:
    val_path = json.load(f)
with open('/Users/freya/Downloads/brain_tumor_image/CNN_model/test_path.json', "r") as f:
    test_path = json.load(f)    
    
label_map = {'G':0, 'M':1, 'N':2, 'P':3}
batch_size = 32

def read_img(img_path):
    img_path = img_path.numpy().decode("utf-8")
    img = Image.open(img_path)
    img = img.convert("RGB")     
    img = np.array(img) / 255.0 # normalize image
    
    index = img_path.rfind('/')+1
    label = img_path[index]
    label_int = label_map[label]
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    label_int = tf.convert_to_tensor(label_int, dtype=tf.int32)
    return img, label_int


def read_img_tf(img_path):
    img, label = tf.py_function(func=read_img,
                                inp=[img_path],
                                Tout=[tf.float32, tf.int32])
    img.set_shape([256, 256, 3]) # img shape
    label.set_shape([])
    return img, label


def augment_image(image, label):
    # adding noise to the image
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
    
    image = tf.image.random_brightness(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.95, upper=1.05)
    
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.005)
    image = image + noise
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label


def generate_dataset(path, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(path)
    dataset = dataset.map(read_img_tf, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(5000)
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset



# for X, y_real in train_dataset.take(1):
#     print(X.shape, y_real.shape)
    
# In[model]
class CNN_model(tf.keras.Model):
    def __init__(self, dropout_rate):
        super().__init__()
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Conv2D(filters=96, kernel_size=30, # set a kernel size bigger enough to find tumor
                                 strides=2, padding='valid'), # shape(batch, length/strides-kernal_size, filters(96))
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),

          tf.keras.layers.Conv2D(filters=128, kernel_size=10, 
                                 strides=2, padding='valid'), 
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),
          
          tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'), 
          tf.keras.layers.Dropout(dropout_rate),


          tf.keras.layers.Conv2D(filters=256, kernel_size=5, 
                                        strides=2, padding='valid', activation='relu'), 
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Conv2D(filters=256, kernel_size=5, 
                                 strides=2, padding='valid', activation='relu'), 
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Conv2D(filters=256, kernel_size=3, 
                                 strides=1, padding='valid', activation='relu'), 
          tf.keras.layers.BatchNormalization(),
          

          tf.keras.layers.GlobalAveragePooling2D(),
          tf.keras.layers.Dense(512, activation='relu'), 
          tf.keras.layers.Dense(126, activation='relu'), 
          tf.keras.layers.Dropout(dropout_rate),
          tf.keras.layers.Dense(4)
        ])
        
    def call(self, x):        
        x = self.seq(x)
        return x
             
            
# In[training setting]
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=40 * 100,
    alpha=0.01
)

    
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.05, clipnorm=1.0)


# early stop: stop training while validation loss is not getting better 
earlystop_callback = tf.keras.callbacks.EarlyStopping(patience=5, 
                                                      mode='min', restore_best_weights=True, 
                                                      start_from_epoch=20)


checkpoint_path = '/Users/freya/Downloads/brain_tumor_image/CNN_model/model_weight.weights.h5'

# check point: save model weight while model is better
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min', # based on accuracy
    save_freq='epoch',
    verbose=1
) # save model while model is better 



# In[training]
Model = CNN_model(dropout_rate = 0.3)

initial_epoch=28 # default is 0, the starting epoch would be n+1 

Model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
    optimizer=optimizer,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# reload model if the model exist
if os.path.exists(checkpoint_path):
    dummy_input = tf.random.normal([batch_size, 256, 256, 3]) # align with image shape
    Model(dummy_input)
    Model.load_weights(checkpoint_path)
    
    
train_dataset = generate_dataset(train_path, batch_size, training=True)
val_dataset = generate_dataset(val_path, batch_size)


Model.fit(train_dataset, epochs=300, validation_data=val_dataset, callbacks=[earlystop_callback, checkpoint_callback], initial_epoch=initial_epoch)


# In[prediction]
test_dataset = generate_dataset(test_path, batch_size)

y_pred = []
y_real = []
for X, y in test_dataset:
    predictions = Model(X, training=False)  
    y_pred.append(predictions.numpy())
    y_real.append(y)
    
y_pred = np.concatenate(y_pred, axis=0)
y_pred = np.argmax(y_pred, axis=1)  

y_real = np.concatenate(y_real, axis=0)

match = y_real==y_pred
print('accuracy: '+str(np.mean(match)))


id_to_label = {0: 'glioma', 1: 'meningioma', 2: 'normal', 3: 'pituitary'}
y_pred_label = [id_to_label[i] for i in y_pred]
y_real_label = [id_to_label[i] for i in y_real]


df = pd.DataFrame({
    "real_label": y_real_label,
    "prediction": y_pred_label
})

df.to_csv('/Users/freya/Downloads/brain_tumor_image/CNN_model/prediction.csv', index=False)


