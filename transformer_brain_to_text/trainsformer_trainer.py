#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 12:35:59 2026

@author: freya
"""
import numpy as np
import tensorflow as tf
from config import freq_count_path, weight_path, cfg_train, cfg_model


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


class LabelSmoothingLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=cfg_model.vocab_size, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def call(self, y_real, y_pred):
        # convert to one-hot
        y_real = tf.cast(y_real, dtype=tf.int64)
        y_true_one_hot = tf.one_hot(y_real, depth=self.num_classes)
        
        # add label smoothing
        smooth_labels = y_true_one_hot * self.confidence + \
                       (1 - y_true_one_hot) * self.smoothing / (self.num_classes - 1)
        
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=smooth_labels,
            logits=y_pred
        )
        
        return tf.reduce_mean(loss)
    
    
# loss with frequency penalty 
def masked_loss(y_real, pred):
  loss_object = LabelSmoothingLoss()
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


# detect current epoch
class EpochTrackerCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, 'current_epoch'):
            self.model.current_epoch.assign(epoch)
            print(f"\n Epoch {epoch + 1} - Scheduled Sampling Stage")
            
            if epoch <= 5:
                print("   Stage 1: noise (10% samples, 10% tokens)")
            elif 5 < epoch <= 10:
                print("   Stage 2: noise (20% samples, 10% tokens)")
            elif 10 < epoch <= 15:
                print("   Stage 3: noise (30% samples, 10% tokens)")
            elif 15 < epoch <= 20:
                print("   Stage 4: noise (40% samples, 10% tokens)")
            else:
                print("   Stage 5: noise (40% samples, 10% tokens)")

epoch_tracker = EpochTrackerCallback()





