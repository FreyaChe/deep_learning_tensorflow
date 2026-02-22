#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 12:55:21 2026

@author: freya
"""
from pathlib import Path
from types import SimpleNamespace

# Paths
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent / 'brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final/'
freq_count_path = base_dir.parent / 'token_freq_counts.npy'
weight_path = base_dir / 'train/L2_acc_005_freq_04.weights.h5'
latest_weight_path = base_dir / 'train/transformer_noise_L3_latest.weights.h5'
gram_path = base_dir.parent.parent / '3-gram.arpa'
csv_path = base_dir /'submission.csv'


# Data
batch_size = [40, 32, 16, 12, 8]


cfg_data = SimpleNamespace()
cfg_data = SimpleNamespace(
    max_ieeg_len = 2500, 
    max_sentence_len = 120,
    num_channels = 512
    )


# Model (Transformer)
cfg_model = SimpleNamespace()
cfg_model = SimpleNamespace(
    num_layers=2, 
    con1_dim=40, 
    conhig_dim=80,
    conmid_dim=64,
    conlow_dim=64,
    data_dim=248,
    higher_kernel=36,
    lower_kernel=6,
    enco_key_dim=128, 
    num_heads=4, 
    enc_dff=768,
    dec_dim=88, 
    dec_key_dim=22, 
    vocab_size=43, 
    sentence_len=149, 
    dec_dff=256, 
    dropout_rate=0.1, 
    att_penalty_score=0.05
    )



'''
num_layers: number of encoder/decoder layers
enco_dim: encode dimention. set as the variances: 512 channels
enco_key_dim: enco_dim/num_heads, for multiheadattention 
dec_dim: dimention after embedding
dec_key_dim: dec_dim/num_heads
vocab_size: the number of output variances
sentence_len: the length of y_real [149]
'''


# Training
cfg_train = SimpleNamespace()
cfg_train = SimpleNamespace(
    lr_dmodel = 1e4,
    epochs = 300,
    early_stop_patience = 10,
    early_stop_start_epoch = 25, 
    freq_penalty_coef = 0.4 #0 means no penalty
    )

