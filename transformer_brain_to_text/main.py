#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 14:19:17 2026

@author: freya
"""

from dataset import get_dataset
from trainsformer_trainer import masked_loss, optimizer, masked_accuracy, earlystop_callback, checkpoint_callback, epoch_tracker
from config import batch_size, weight_path, cfg_model, cfg_train, latest_weight_path
from transformer_model import generate_transformer


# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# initialize
initial_epoch=0
transformer = generate_transformer(cfg_model, weight_path, initial_epoch=initial_epoch)
    
# get dataset
train_dataset = get_dataset('train', batch_size)
val_dataset = get_dataset('valid', batch_size)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)
# ,steps_per_execution=4

transformer.fit(train_dataset, epochs=cfg_train.epochs, validation_data=val_dataset, 
                        callbacks=[earlystop_callback, checkpoint_callback, epoch_tracker], initial_epoch=initial_epoch)

transformer.save_weights(latest_weight_path)
    




