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
import numpy as np


def trial_generator(file_list, training):
    for file_path in file_list:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                g = f[key]
                neural_features = g['input_features'][:]
                
                # per-channel  norm
                ch_min = neural_features.min(axis=0, keepdims=True)
                ch_max = neural_features.max(axis=0, keepdims=True)
                den = ch_max - ch_min
                den[den == 0] = 1
                norm_features = (neural_features - ch_min) / den

                # 处理单个trial
                if training:
                    chan_std = neural_features.std(axis=0)
                    noise = np.random.normal(0, 0.02, neural_features.shape) * chan_std
                    norm_features = np.clip(norm_features + noise, 0, 1)
       
                # 处理序列
                seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
                if seq_class_ids is not None:
                    middle_list = seq_class_ids[seq_class_ids != 0].tolist()
                    y = [41] + middle_list + [42]
                else:
                    y = [41, 42]
                
                # # Padding y
                # if len(y) < cfg_data.max_sentence_len:
                #     y = y + [0] * (cfg_data.max_sentence_len - len(y))
                # else:
                #     y = y[:cfg_data.max_sentence_len]
                
                decoder_input = y[:-1]
                target_output = y[1:]
                
                # Yield单个样本
                yield (norm_features, decoder_input), target_output



def generate_dataset(file_list, batch_size, training=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: trial_generator(file_list, training),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, cfg_data.num_channels), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int64)
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)
        )
    )
    
    if training:
        dataset = dataset.shuffle(buffer_size=100) 
    
    bucket_boundaries = [630, 780, 930, 1150]
    bucket_batch_sizes = batch_size

    def key_func(x_y, target):
        length = tf.shape(x_y[0])[0]
        boundaries = tf.constant(bucket_boundaries, dtype=tf.int32)
        bucket_id = tf.reduce_sum(tf.cast(length >= boundaries, tf.int32))
        return tf.cast(bucket_id, tf.int64)
    
    def reduce_func(key, windowed_dataset):
        key = tf.cast(key, tf.int32)
        # 根据 bucket key 选择对应的 batch size
        batch_size_options = tf.constant(bucket_batch_sizes, dtype=tf.int64)
        current_batch_size = batch_size_options[key]
        
        return windowed_dataset.padded_batch(
            current_batch_size,
            padding_values=(
                (tf.constant(0.0), tf.cast(0, tf.int64)),
                tf.cast(0, tf.int64)
            ),
            drop_remainder=True
        )
    
    # window_size 设成最大的 batch size，保证桶里有足够的数据
    max_batch = max(bucket_batch_sizes)
    dataset = dataset.group_by_window(
        key_func=key_func,
        reduce_func=reduce_func,
        window_size=max_batch
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset



# def check_bucket_distribution(file_list):
#     boundaries = [630, 780, 930, 1150]
#     counts = [0] * (len(boundaries) + 1)
    
#     for file_path in file_list:
#         with h5py.File(file_path, 'r') as f:
#             for key in f.keys():
#                 l = min(f[key]['input_features'].shape[0], cfg_data.max_ieeg_len)
#                 for i, b in enumerate(boundaries):
#                     if l < b:
#                         counts[i] += 1
#                         break
#                 else:
#                     counts[-1] += 1
    
#     labels = ['<630', '630-780', '780-930', '950-1150', '>1150']
#     for label, count in zip(labels, counts):
#         print(f"{label}: {count} samples ({count/sum(counts)*100:.1f}%)")

# file_dir = glob.glob(os.path.join(data_dir, "**", "*.hdf5"), recursive=True)
# check_bucket_distribution(file_dir)

# def check_target_distribution(file_list):
#     lengths = []
#     N_len = []
#     for file_path in file_list:
#         with h5py.File(file_path, 'r') as f:
#             for key in f.keys():
#                 g = f[key]
#                 neural_features = g['input_features'][:]
#                 N_len.append(len(neural_features))
#                 if 'seq_class_ids' in g:
#                     ids = g['seq_class_ids'][:]
#                     real_len = len(ids[ids != 0]) + 2  # +2 是 start/end token
#                     lengths.append(real_len)
    
#     lengths = np.array(lengths)
#     N_len = np.array(N_len)
#     print(f"target 最短: {lengths.min()}")
#     print(f"target 最长: {lengths.max()}")
#     print(f"target 均值: {lengths.mean():.0f}")
#     print(f"target 占 max_sentence_len 的比例: {lengths.mean()/cfg_data.max_sentence_len:.1%}")
#     print(f"Neuro 最短: {N_len.min()}")
#     print(f"Neuro 最长: {N_len.max()}")
#     print(f"Neuro 均值: {N_len.mean():.0f}")
#     print(f"Neuro 占 max_sentence_len 的比例: {N_len.mean()/2500:.1%}")


# file_dir = glob.glob(os.path.join(data_dir, "**", "*_val.hdf5"), recursive=True)
# check_target_distribution(file_dir)


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
        
