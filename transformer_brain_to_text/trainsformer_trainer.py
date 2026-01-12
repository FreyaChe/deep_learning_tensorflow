#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 12:35:59 2026

@author: freya
"""
import numpy as np
import tensorflow as tf
from config import freq_count_path, weight_path, cfg_train


# optimizer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=2000):
    super().__init__()
    self.d_model = tf.cast(d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model=cfg_train.lr_dmodel)
optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.05, clipnorm=1.0)


# load frequency counts for frequency penalty
counts_np = np.load(freq_count_path)
freqs = tf.convert_to_tensor(counts_np)


# loss with frequency penalty 
def masked_loss(y_real, pred):
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  
  loss = loss_object(y_real, pred)
  mask = tf.logical_and(tf.not_equal(y_real, 41), tf.not_equal(y_real, 0))
  loss = tf.cast(loss, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  
  y_real = tf.cast(y_real, dtype=tf.int64)
  weights_tensor = tf.gather(freqs, y_real)  # [batch, seq_len]
  loss = loss * (1.0 + cfg_train.freq_penalty_coef * (weights_tensor - 0.5))    # [batch, seq_len]
  
  loss *= mask
  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


# accuracy
def masked_accuracy(y_real, pred):
  pred = tf.argmax(pred, axis=2)
  match = y_real == pred
  mask = tf.logical_and(tf.not_equal(y_real, 41), tf.not_equal(y_real, 0))
  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)



# early stop: stop training while validation loss is not getting better 
earlystop_callback = tf.keras.callbacks.EarlyStopping(patience=cfg_train.early_stop_patience, 
                                                      mode='min', restore_best_weights=True, 
                                                      start_from_epoch=cfg_train.early_stop_start_epoch)



# check point: save model weight while model is better
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=weight_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min', # based on accuracy
    save_freq='epoch',
    verbose=1
) # save model while model is better 



def train(model, train_dataset, val_dataset, cfg_train):
    model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy]
    )

    history = model.fit(train_dataset, epochs=cfg_train.epochs, validation_data=val_dataset, 
                        callbacks=[earlystop_callback, checkpoint_callback])
    return history


