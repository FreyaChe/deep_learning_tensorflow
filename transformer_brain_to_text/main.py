#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 14:19:17 2026

@author: freya
"""

from dataset import get_dataset
from trainsformer_trainer import train
from config import batch_size, weight_path, cfg_model, cfg_train
from transformer_model import generate_transformer


def main():
    # initialize
    transformer = generate_transformer(cfg_model, weight_path)
    
    # get dataset
    train_dataset = get_dataset('train', batch_size)
    val_dataset = get_dataset('valid', batch_size)

    train(
        model=transformer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg_train=cfg_train
    )

if __name__ == "__main__":
    main()


