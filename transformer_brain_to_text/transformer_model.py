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
from config import cfg_data
import os


# In[create elements]
class BaseAttention(tf.keras.layers.Layer): # initialize for attention
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class FeedForward(tf.keras.layers.Layer): # feedforward for both encoder and decoder 
  def __init__(self, enco_dim, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(enco_dim),
      tf.keras.layers.Dropout(dropout_rate)
    ]) # shape(batch, time_length:2500, enco_dim)
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x



# In[encoder]
class GlobalSelfAttention(BaseAttention): # for input, with all input items 
  def call(self, x, mask):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        attention_mask=mask) 
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x


def positional_encoding(length, depth): 
  # depth can be >64 and <512, shall able to be devided by num_heads
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 
  return tf.cast(pos_encoding, dtype=tf.float32)


class Encodeinput(tf.keras.layers.Layer): # for ieeg data 
    def __init__(self, enco_dim):
        super(Encodeinput, self).__init__()
        self.enco_dim  = enco_dim
        # conv1D
        self.conv1d     = tf.keras.layers.Conv1D(filters=enco_dim, kernel_size=6, 
                                                 strides=2, padding='same', activation='relu')
        self.pos_encoding = positional_encoding(length=int(cfg_data.max_ieeg_len / 2), depth=enco_dim) # length: based on the strides

    def call(self, x):        
        mask = tf.math.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        mask = mask[:, ::2]
        x = self.conv1d(x)
        output_length = tf.shape(x)[1]
        mask = mask[:, :output_length]
        mask = tf.cast(mask, dtype=tf.bool)
        x *= tf.math.sqrt(tf.cast(self.enco_dim, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :output_length, :]
        return x, mask


class EncoderLayer(tf.keras.layers.Layer):
  '''
  including one maltihead attention and one feedforword
  '''
  def __init__(self,*, key_dim, enco_dim, num_heads, dff, dropout_rate=0.1): 
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=key_dim, # Size of each attention head for query and key（input_dim/num_heads） 
        dropout=dropout_rate)

    self.ffn = FeedForward(enco_dim, dff)

  def call(self, x, mask):
    x = self.self_attention(x, mask) # shape(batch, time_len:2500, variance:512)
    x = self.ffn(x) 
    return x


class Encoder(tf.keras.layers.Layer):
  '''
    including one encode input and encode layers
    num_layers: number of encoder layers, 
    conv_dim: the dimention of conv1D output,
    num_heads: head of attention
  '''  
  def __init__(self, *, num_layers, enco_dim, key_dim, num_heads,
               dff, dropout_rate=0.1): 
    super().__init__()

    self.enco_dim = enco_dim
    self.num_layers = num_layers # number of encoder layer

    self.enc_input = Encodeinput(
        enco_dim) 

    self.enc_layers = [
        EncoderLayer(key_dim=key_dim,
                     enco_dim=enco_dim,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    x,mask = self.enc_input(x)  # Shape(batch_size, strided len:2500/2, enco_dim)
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
   mask = mask[:, tf.newaxis, :] & mask[:, :, tf.newaxis]
   attn_output = self.mha(
       query=x,
       value=x,
       key=x,
       attention_mask=mask,
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

    def call(self, x):
        length = tf.shape(x)[1]
        mask = self.embedding.compute_mask(x)
        mask = tf.cast(mask, dtype=tf.int64)
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dec_dim, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
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
  def __init__(self, *, num_layers, enco_dim, enco_key_dim, num_heads, enc_dff,
               dec_dim, dec_key_dim, vocab_size, sentence_len, dec_dff, dropout_rate=0.1, att_penalty_score=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, enco_dim=enco_dim,
                           key_dim=enco_key_dim, num_heads=num_heads, dff=enc_dff,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, dec_dim=dec_dim, num_heads=num_heads, 
                           key_dim=dec_key_dim, vocab_size=vocab_size, sentence_len=sentence_len, 
                           dff=dec_dff, dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(vocab_size)
    self.penalty_score = att_penalty_score
    
  def call(self, inputs):
    X, y = inputs # query=dec_input, context=enc_output
    X, enc_mask = self.encoder(X)  # (batch_size, strided len:2500/2, enco_dim) mask:(batch, strided length)
    y, cross_mask = self.decoder(y, X, enc_mask)  # (batch_size, vocab_size, dec_dim)
    logits = self.final_layer(y)  # (batch_size, vocab_size, target_vocab_size)
    
    # add decoder attention penalty to makes model more rely on EEG encoder
    attn_scores = self.decoder.last_attn_scores # shape(batch, num_heads, sentence_length, strided length)
    mask = tf.cast(cross_mask[:, tf.newaxis, :, :], tf.float32)
    entropy = -tf.reduce_sum(
    attn_scores * tf.math.log(attn_scores + 1e-8) * mask,
    axis=-1
    )  # (batch, heads, dec_len)
    penalty = tf.reduce_mean(entropy)
    self.add_loss(self.penalty_score * penalty) # add_loss add the number to masked loss 
    return logits


# In[initialize and reload weight]
def generate_transformer(cfg_model, weight_path):
    # initialize
    transformer = Transformer(num_layers=cfg_model.num_layers, 
                              enco_dim=cfg_model.enco_dim, 
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
        dummy_enc = tf.zeros((1, cfg_data.max_ieeg_len, cfg_data.num_channels))
        dummy_dec = tf.zeros((1, cfg_data.max_sentence_len))
        _ = transformer((dummy_enc, dummy_dec), training=False)
        transformer.load_weights(weight_path)
    return transformer
    




