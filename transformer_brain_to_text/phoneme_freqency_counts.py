#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 12:41:34 2026

phonemes frequency count to create penalty for loss 
(avoide model use the offenly occured phonemes)

@author: freya
"""

import tensorflow as tf
import glob
import os
import numpy as np
from dataset import generate_dataset
from config import data_dir, freq_count_path, batch_size


train_file_dir = glob.glob(os.path.join(data_dir, "**", "*_train.hdf5"), recursive=True)
train_dataset = generate_dataset(train_file_dir, batch_size, training=True)
valid_file_dir = glob.glob(os.path.join(data_dir, "**", "*_val.hdf5"), recursive=True)
val_dataset = generate_dataset(valid_file_dir, batch_size)


total_counts = tf.zeros([43], dtype=tf.int64)

def normalize(x):
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    return (x - x_min) / (x_max - x_min + 1e-8)


for batch, (inp, targ) in enumerate(train_dataset):
    batch = tf.reshape(targ, [-1])  # convert to 1D
    batch_counts = tf.math.bincount(
        batch,
        minlength=43,
        maxlength=43,
        dtype=tf.int64
    )
    total_counts += batch_counts
    
total_counts = tf.zeros([43], dtype=tf.int64)
for batch, (inp, targ) in enumerate(val_dataset):
    batch = tf.reshape(targ, [-1])  # convert to 1D
    batch_counts = tf.math.bincount(
        batch,
        minlength=43,
        maxlength=43,
        dtype=tf.int64
    )
    total_counts += batch_counts

tt = tf.cast(total_counts, tf.float32).numpy()
tt[0], tt[40] = 0.5, 0.5 # make sure the break, padding 
tt[40] = max(tt)
counts = normalize(tt).numpy()
counts[0], counts[41] = 0.5, 0.5
np.save(freq_count_path, counts)

