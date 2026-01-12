#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 12:27:38 2026

convert data to dataset and add noise to training data

example:
batch_size = 64
root_dir = 'data_path'
train_file_dir = glob.glob(os.path.join(root_dir, "**", "*_train.hdf5"), recursive=True)
train_dataset = generate_dataset(train_file_dir, batch_size, training=True)

@author: freya
"""

import tensorflow as tf
import h5py
import glob
import os
from config import data_dir, cfg_data


# read raw data 
def read_hdf5_file(file_path, training):        
    # tensor to str
    file_path = file_path.numpy()
    inputs = []
    outputs = []
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]
            neural_features = g['input_features'][:]
            # add noise into iEEG input
            chan_std = neural_features.std(axis=0)
            if training:
                noise = tf.random.normal(tf.shape(neural_features), mean = 0.0, stddev=0.02)
                noise = noise*chan_std
                neural_features = neural_features + noise.numpy()
            den = neural_features.max() - neural_features.min()
            if den == 0:
                den = 1e-9
            norm_features = (neural_features - neural_features.min()) / den
            inputs.append(norm_features)

            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None  
            if seq_class_ids is not None:
                middle_list = seq_class_ids[seq_class_ids != 0].tolist()
                outputs.append([41] + middle_list + [42])     
            else:
                outputs.append([41, 42])
        X = tf.keras.utils.pad_sequences(inputs,maxlen=cfg_data.max_ieeg_len, padding="post", dtype="float32") # padding data
        y_real = tf.keras.utils.pad_sequences(outputs,maxlen=cfg_data.max_sentence_len, padding="post", dtype='int64') # padding data
    return X, y_real


# convert data to tensor formate
def make_tf_read_hdf5(training):
    def _tf_read_hdf5(file_path):
        X, y = tf.py_function(
        func=read_hdf5_file,
        inp=[file_path, training],
        Tout=[tf.float32, tf.int64]
        )
        X.set_shape([None, cfg_data.max_ieeg_len, cfg_data.num_channels])
        y.set_shape([None, cfg_data.max_sentence_len])  
        decoder_input = y[:, :-1]
        target_output = y[:, 1:] 
        decoder_input.set_shape([None, cfg_data.max_sentence_len-1])
        target_output.set_shape([None, cfg_data.max_sentence_len-1])
        return tf.data.Dataset.from_tensor_slices(((X, decoder_input), target_output))
    return _tf_read_hdf5

# create Dataset
def generate_dataset(file_list, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    
    if training:
        dataset = dataset.shuffle(len(file_list)) # shuffle file dir
    
    dataset = dataset.interleave(
         make_tf_read_hdf5(training),
         cycle_length=2, #tf.data.AUTOTUNE,
         num_parallel_calls=tf.data.AUTOTUNE,
         deterministic=not training
    )
       
    # shuffel data
    if training:
        dataset = dataset.shuffle(buffer_size=500) # shuffle data 

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) # preload data to accelerate speed 
    return dataset



# get dataset
def get_dataset(data_type, batch_size):
    if data_type=='train':
        data_name = "*_train.hdf5"
        training = True
    elif data_type=='valid':
        data_name = "*_val.hdf5"
        training = False
    elif data_type=='test':
        data_name = "*_test.hdf5"
        training = False
    else:
        raise ValueError(f"Unknown data_type: {data_type}, data_type should from train, valid, test")
    file_dir = glob.glob(os.path.join(data_dir, "**", data_name), recursive=True)
    dataset = generate_dataset(file_dir, batch_size, training=training)
    return dataset
        
