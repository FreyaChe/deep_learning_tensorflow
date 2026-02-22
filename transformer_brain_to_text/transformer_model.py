#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 12:08:56 2026

transformer model structure:
encoder:
    encode input: Conv1D, positional encoding
    encode layer: glabal attention, feedforward
decoder:
    decode embedding: embedding, positional encoding
    decode layer: causal self attention, cross attention, feedforward
final layer:
    dense
    

@author: freya
"""

import tensorflow as tf
import numpy as np
from config import cfg_data, batch_size
import os


# In[create elements]
class BaseAttention(tf.keras.layers.Layer): # initialize for attention
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()



class FeedForward(tf.keras.layers.Layer): # feedforward for both encoder and decoder 
  def __init__(self, output_dim, squeeze_dim, dropout_rate=0.1):
    super().__init__()
    
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(squeeze_dim, activation='relu'),
      tf.keras.layers.Dense(output_dim),
      tf.keras.layers.Dropout(dropout_rate)
    ]) # shape(batch, time_length:2500, output_dim: shall equal to enco_dim)
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.layer_norm(x)
    x = self.add([x, self.seq(x)]) 
    return x



def add_dec_noise(decoder_input, ratio=0.1, percent=0.1):
    '''
    ratio is the percentage of noise channels 
    percent is the percentage of location in channels is randomed 
    randomed number is from 1-40, avoided padding, start and end
    '''
    seq_len = tf.shape(decoder_input)[1]
    batch_size = tf.shape(decoder_input)[0]
    
    num_noisy = tf.cast(tf.cast(batch_size, tf.float32) * ratio, tf.int32)
    indices = tf.random.shuffle(tf.range(batch_size))
    noisy_indices = indices[:num_noisy]
    channel_idx = tf.reduce_any(
        tf.equal(tf.range(batch_size)[:, tf.newaxis], noisy_indices[tf.newaxis, :]),
        axis=1
    )
    
    valid_mask = tf.logical_and(
            tf.not_equal(decoder_input, 0),
            tf.logical_and(
                tf.not_equal(decoder_input, 41),
                tf.not_equal(decoder_input, 42)
            )
        )
    
    token_mask = tf.random.uniform([batch_size, seq_len]) < percent
    #combine all of masks
    final_mask = tf.logical_and(
        channel_idx[:, tf.newaxis],  
        tf.logical_and(valid_mask, token_mask)
    )  # (batch, seq_len)

    random_tokens = tf.random.uniform(
        [batch_size, seq_len],
        minval=1,
        maxval=40,
        dtype=tf.int64
    )
    
    noisy_input = tf.where(final_mask, random_tokens, decoder_input)
    return noisy_input
    


# In[encoder]
class GlobalSelfAttention(BaseAttention): # for input, with all input items 
  def call(self, x, mask):
    x = self.layernorm(x)
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        attention_mask=mask) 
    x = self.add([x, attn_output])
    return x



def positional_encoding(length, depth): 
  depth = depth//2
  depth = tf.cast(depth, dtype=tf.int64)
  
  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 
  return tf.cast(pos_encoding, dtype=tf.float32)



class Encodeinput(tf.keras.layers.Layer): # for ieeg data 
    def __init__(self, *, con1_dim, conhig_dim, conmid_dim, conlow_dim, data_dim,
                 higher_kernel, lower_kernel, dropout_rate=0.1):
        super(Encodeinput, self).__init__()
        # conv1D
        self.con_1 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=con1_dim, kernel_size=2, 
                                                     strides=2, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU()
            ])
        
        self.con_high = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=conhig_dim, kernel_size=higher_kernel, 
                                                     strides=1, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')    
            ])
        
        self.con_mid = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=conmid_dim, kernel_size=higher_kernel//3, 
                                                     strides=1, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')    
            ])
        
        self.con_low = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=conlow_dim, kernel_size=lower_kernel, 
                                                     strides=1, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')
            ])
        
        self.data_dim = data_dim
        self.mask_convert = tf.keras.layers.MaxPool1D(pool_size=higher_kernel, strides=2, padding='same')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.proj = tf.keras.layers.Dense(self.data_dim, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        self.pos_encoding = positional_encoding(length=cfg_data.max_ieeg_len, depth=self.data_dim) # length: based on the strides
        self.pos_scale = tf.Variable(0.1, trainable=True, dtype=tf.float32)
        
    def call(self, x):        
        mask = tf.math.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        x_1 = self.con_1(x)
        x_high = self.con_high(x) # conv with different kernel lenth
        x_mid = self.con_mid(x) 
        x_low = self.con_low(x) 
        x = tf.concat([x_high, x_mid, x_low, x_1], axis=-1) # concat two kernal result in one matrix [batch_size, EEG_length, time_dim*2]
        x = self.layer_norm(x)
        x = self.proj(x)        
        x = self.layer_norm2(x)
        x = self.dropout(x)
        
        # convert mask based on the kernel size
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, -1)
        mask = self.mask_convert(mask)
        mask = tf.squeeze(mask, -1)
        mask = tf.cast(mask, tf.bool)
        
        seq_len = tf.shape(x)[1]
        # x = x + self.pos_encoding[tf.newaxis, :seq_len, :]
        x = x + self.pos_scale * self.pos_encoding[tf.newaxis, :seq_len, :]
        # mask_expanded = tf.expand_dims(mask, axis=-1)
        # x = x * tf.cast(mask_expanded,tf.float32)
        return x, mask


class EncoderLayer(tf.keras.layers.Layer):
  '''
  including one maltihead attention and one feedforword
  '''
  def __init__(self,*, key_dim, data_dim, num_heads, dff, dropout_rate=0.1): 
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=key_dim, # Size of each attention head for query and key（input_dim/num_heads） 
        dropout=dropout_rate)

    self.ffn = FeedForward(data_dim, dff) 

  def call(self, x, mask):
    x = self.self_attention(x, mask) # shape(batch, time_len:2500, variance:400)
    x = self.ffn(x) 
    return x


class Encoder(tf.keras.layers.Layer):
  '''
    including one encode input and encode layers
    num_layers: number of encoder layers, 
    conv_dim: the dimention of conv1D output,
    num_heads: head of attention
  '''  
  def __init__(self, *, num_layers, con1_dim, conhig_dim, conmid_dim, conlow_dim, data_dim,
               higher_kernel, lower_kernel, key_dim, num_heads, dff, dropout_rate=0.1): 
    super().__init__()
    self.num_layers = num_layers # number of encoder layer

    self.enc_input = Encodeinput(
        con1_dim=con1_dim, 
        conhig_dim=conhig_dim,
        conmid_dim=conmid_dim,
        conlow_dim=conlow_dim,
        data_dim=data_dim,
        higher_kernel=higher_kernel, 
        lower_kernel=lower_kernel, 
        dropout_rate=dropout_rate) 

    self.enc_layers = [
        EncoderLayer(key_dim=key_dim,
                     data_dim=data_dim,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    x, mask = self.enc_input(x)  # Shape(batch_size, strided len:2500/2, enco_dim)
    mask = tf.cast(mask, tf.int64)
    atten_mask = mask[:, tf.newaxis, :] & mask[:, :, tf.newaxis]
    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, atten_mask)

    return x, mask  # Shape(batch_size, strided len:2500/2, enco_dim)



# In[decoder]
class CausalSelfAttention(BaseAttention):  # for output, mask the padding item
  def call(self, x, mask):
   # mask = mask[:, tf.newaxis, :] & mask[:, :, tf.newaxis]
   padding_mask = mask[:, tf.newaxis, :]
   attn_output = self.mha(
       query=x,
       value=x,
       key=x,
       attention_mask=padding_mask,
       use_causal_mask=True)
   x = self.add([x, attn_output])
   x = self.layernorm(x)
   return x


class CrossAttention(BaseAttention): # for input, output connection
  def call(self, x, context, cross_mask):
    attn_output, attn_scores = self.mha(
        query=x, # query=dec_input, context=enc_output
        key=context,
        value=context,
        attention_mask=cross_mask,
        return_attention_scores=True)

    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x


class DecodeEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, sentence_len, dec_dim):
        super(DecodeEmbedding, self).__init__()
        self.dec_dim  = dec_dim # embedding unit
        self.embedding = tf.keras.layers.Embedding(vocab_size, dec_dim, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=sentence_len, depth=dec_dim) # based on the strides
        self.pos_scale = tf.Variable(0.1, trainable=True, dtype=tf.float32)

    def call(self, x):
        length = tf.shape(x)[1]
        mask = self.embedding.compute_mask(x)
        mask = tf.cast(mask, dtype=tf.int64)
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dec_dim, tf.float32))
        # x = x + self.pos_encoding[tf.newaxis, :length, :]
        x = x + self.pos_scale * self.pos_encoding[tf.newaxis, :length, :]

        # mask_expanded = tf.expand_dims(mask, axis=-1)
        # x = x * tf.cast(mask_expanded,tf.float32)
        return x, mask


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, *, dec_dim, key_dim, num_heads, dff, dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate) 

    self.ffn = FeedForward(dec_dim, dff, dropout_rate)

  def call(self, x, context, dec_mask, cross_mask):
    x = self.causal_self_attention(x=x,mask=dec_mask) # (batch_size, sentence_len, dec_dim)
    x = self.cross_attention(x=x, context=context, cross_mask=cross_mask) # (batch_size, sentence_len, dec_dim)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores # (batch_size, sentence_len:150, dec_dim)

    x = self.ffn(x)  # (batch_size, sentence_len:150, dec_dim)
    return x


class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, dec_dim, key_dim, vocab_size, sentence_len, num_heads, 
               dff, dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.dec_dim = dec_dim
    self.num_layers = num_layers

    self.dec_embedding = DecodeEmbedding(vocab_size=vocab_size, sentence_len=sentence_len,
                                             dec_dim=dec_dim)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(dec_dim=dec_dim, key_dim=key_dim, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context, enc_mask): # query=dec_input, context=enc_output
    x, dec_mask = self.dec_embedding(x)  # x:(batch_size, vocab_size, dec_dim) mask:(batch, sent_lenth)
    cross_mask = dec_mask[:, :,tf.newaxis] & enc_mask[:, tf.newaxis, :] # shape(batch,query:x=dec_input,value:context=enc_output)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context, dec_mask, cross_mask)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores
    return x, cross_mask #shape(batch_size, vocab_size, dec_dim)



# In[transformer]
class Transformer(tf.keras.Model):
    def __init__(self, *, batch_size, num_layers, con1_dim, conhig_dim, conmid_dim, 
                 conlow_dim, data_dim, higher_kernel, lower_kernel, enco_key_dim, 
                 num_heads, enc_dff, dec_dim, dec_key_dim, vocab_size, sentence_len, 
                 dec_dff, dropout_rate=0.1, att_penalty_score=0.1):
        
      super().__init__()
      self.encoder = Encoder(num_layers=num_layers, con1_dim=con1_dim, 
                             conhig_dim=conhig_dim, conmid_dim=conmid_dim,
                             conlow_dim=conlow_dim, data_dim=data_dim, 
                             higher_kernel=higher_kernel,
                             lower_kernel=lower_kernel, key_dim=enco_key_dim, 
                             num_heads=num_heads, dff=enc_dff,
                             dropout_rate=dropout_rate)
      
      self.decoder = Decoder(num_layers=num_layers, dec_dim=dec_dim, num_heads=num_heads, 
                             key_dim=dec_key_dim, vocab_size=vocab_size, sentence_len=sentence_len, 
                             dff=dec_dff, dropout_rate=dropout_rate)

      self.final_layer = tf.keras.layers.Dense(vocab_size)
      self.penalty_score = att_penalty_score
      self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
      self.vocab_size = vocab_size
      
    def call(self, inputs, training=False):
        X,y = inputs
        X, enc_mask = self.encoder(X)  # (batch_size, strided len:2500/2, enco_dim) mask:(batch, strided length)

        if training:
            epoch = self.current_epoch
            y = tf.cond(
               epoch <= 5,
               lambda: add_dec_noise(y, ratio=0.1, percent=0.1),
               lambda: tf.cond(
                   epoch <= 10,
                   lambda: add_dec_noise(y, ratio=0.2, percent=0.1),
                   lambda: tf.cond(
                       epoch <= 15,
                       lambda: add_dec_noise(y, ratio=0.3, percent=0.1),
                       lambda: tf.cond(
                           epoch <= 20,
                           lambda: add_dec_noise(y, ratio=0.4, percent=0.1),
                           lambda: add_dec_noise(y, ratio=0.4, percent=0.1)
                       )
                   )
               )
           )
        
        y, cross_mask = self.decoder(y, X, enc_mask)  # (batch_size, vocab_size, dec_dim)
        logits = self.final_layer(y)
                
        # add decoder attention penalty to makes model more rely on EEG encoder
        attn_scores = self.decoder.last_attn_scores # shape(batch, num_heads, sentence_length, strided length)
        mask = tf.cast(cross_mask[:, tf.newaxis, :, :], tf.float32)
        entropy = -tf.reduce_sum(
        attn_scores * tf.math.log(attn_scores + 1e-8) * mask,
        axis=-1
        )  # (batch, heads, dec_len)
        penalty = -tf.reduce_mean(entropy)
        penalty = tf.clip_by_value(penalty, -2.0, 0.0)
        self.add_loss(self.penalty_score * penalty) # add_loss add the number to masked loss 
        return logits


# In[initialize and reload weight]
def generate_transformer(cfg_model, weight_path,initial_epoch=0):
    # initialize
    transformer = Transformer(batch_size = batch_size,
                              num_layers=cfg_model.num_layers, 
                              con1_dim=cfg_model.con1_dim, 
                              conhig_dim=cfg_model.conhig_dim,
                              conmid_dim=cfg_model.conmid_dim,
                              conlow_dim=cfg_model.conlow_dim,
                              data_dim=cfg_model.data_dim,
                              higher_kernel=cfg_model.higher_kernel,
                              lower_kernel=cfg_model.lower_kernel, 
                              enco_key_dim=cfg_model.enco_key_dim, 
                              num_heads=cfg_model.num_heads, 
                              enc_dff=cfg_model.enc_dff,
                              dec_dim=cfg_model.dec_dim, 
                              dec_key_dim=cfg_model.dec_key_dim, 
                              vocab_size=cfg_model.vocab_size, 
                              sentence_len=cfg_model.sentence_len, 
                              dec_dff=cfg_model.dec_dff, 
                              dropout_rate=cfg_model.dropout_rate, 
                              att_penalty_score=cfg_model.att_penalty_score)
    
    # reload model if the model exist
    if os.path.exists(weight_path):
        transformer.current_epoch.assign(initial_epoch)
        dummy_enc = tf.zeros((1, cfg_data.max_ieeg_len, cfg_data.num_channels))
        dummy_dec = tf.zeros((1, cfg_model.sentence_len))
        _ = transformer((dummy_enc, dummy_dec), training=False)
        transformer.load_weights(weight_path)
    return transformer
    




