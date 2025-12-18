#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 16:55:07 2025

@author: freya
"""

import tensorflow as tf
import h5py
import glob
import os
import numpy as np


# In[make data generator]
# read raw data 
def read_hdf5_file(file_path):        
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
            den = neural_features.max() - neural_features.min()
            if den == 0:
                den = 1e-9
            norm_features = (neural_features - neural_features.min()) / den
            inputs.append(norm_features)

            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None  
            middle_list = seq_class_ids[seq_class_ids != 0].tolist()
            outputs.append([41] + middle_list + [42])            
        
        X = tf.keras.utils.pad_sequences(inputs,maxlen=2500, padding="post", dtype="float32") # padding data
        y_real = tf.keras.utils.pad_sequences(outputs,maxlen=150, padding="post", dtype='int64') # padding data
    return X, y_real


# convert data to tensor formate
def tf_read_hdf5(file_path):
    X, y = tf.py_function(
    func=read_hdf5_file,
    inp=[file_path],
    Tout=[tf.float32, tf.int64]
    )
    X.set_shape([None, 2500, 512])
    y.set_shape([None, 150])  
    decoder_input = y[:, :-1]
    target_output = y[:, 1:] 
    decoder_input.set_shape([None, 149])
    target_output.set_shape([None, 149])
    return tf.data.Dataset.from_tensor_slices(((X, decoder_input), target_output))


# create Dataset
def generate_dataset(file_list, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    
    if training:
        dataset = dataset.shuffle(len(file_list)) # shuffle file dir
    
    dataset = dataset.interleave(
         tf_read_hdf5,
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



# In[create elements]
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


class EndocdeEmbedding(tf.keras.layers.Layer): # for eeg data 
    def __init__(self, enco_dim):
        super(EndocdeEmbedding, self).__init__()
        self.enco_dim  = enco_dim
        # conv1D
        self.conv1d     = tf.keras.layers.Conv1D(filters=enco_dim, kernel_size=6, 
                                                 strides=2, padding='same', activation='relu')
        self.pos_encoding = positional_encoding(length=int(2500 / 2), depth=enco_dim) # length: based on the strides

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
        
    
# embed_input = EndocdeEmbedding(conv_dim=300) #256
# t1, mask = embed_input(tt) # t1 shape (batch, strided length: 2500/2, conv_dim: 300)

# decod_output = DecodeEmbedding(input_dim=150, output_dim=10)
# t1, mask = decod_output(tt) # t1 shape (batch, input_dim:150, output_dim:10)

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class FeedForward(tf.keras.layers.Layer):
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
    including one embedding and encode layer
    num_layers: number of encoder layers, 
    conv_dim: the dimention of conv1D output,
    num_heads: head of attention,
  '''  
  def __init__(self, *, num_layers, enco_dim, key_dim, num_heads,
               dff, dropout_rate=0.1): 
    super().__init__()

    self.enco_dim = enco_dim
    self.num_layers = num_layers # number of encoder layer

    self.enc_embedding = EndocdeEmbedding(
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
    x,mask = self.enc_embedding(x)  # Shape(batch_size, strided len:2500/2, enco_dim)
    mask = mask[:, tf.newaxis, :] & mask[:, :, tf.newaxis]
    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, mask)

    return x  # Shape(batch_size, strided len:2500/2, enco_dim)


# encoder = Encoder(num_layers=1, enco_dim=512, key_dim=128, num_heads=4, dff=1000, dropout_rate=0.1)
# enco_output = encoder(X[1:3,:,:])


# In[decoder]
class CausalSelfAttention(BaseAttention):  # for output, mask the later item
  def call(self, x, mask):
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
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x, # query=dec_input, context=enc_output
        key=context,
        value=context,
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
        mask = tf.cast(mask, dtype=tf.bool)
        
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
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

    self.ffn = FeedForward(dec_dim, dff)

  def call(self, x, context, mask):
    x = self.causal_self_attention(x=x,mask=mask) # (batch_size, sentence_len, dec_dim)
    x = self.cross_attention(x=x, context=context) # (batch_size, sentence_len, dec_dim)

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

  def call(self, x, context): # query=dec_input, context=enc_output
    x, mask = self.dec_embedding(x)  # (batch_size, vocab_size, dec_dim)
    mask = mask[:, tf.newaxis, :] & mask[:, :, tf.newaxis]

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context, mask)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores
    return x #shape(batch_size, vocab_size, dec_dim)


# decoder = Decoder(num_layers=1, dec_dim=88, num_heads=4, key_dim=22, vocab_size=43, sentence_len=150, dff=400, dropout_rate=0.1)
# deco_output = decoder(tt, enco_output) #(batch_size, vocab_size, d_model)



# In[transformer]
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, enco_dim, enco_key_dim, num_heads, enc_dff,
               dec_dim, dec_key_dim, vocab_size, sentence_len, dec_dff, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, enco_dim=enco_dim,
                           key_dim=enco_key_dim, num_heads=num_heads, dff=enc_dff,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, dec_dim=dec_dim, num_heads=num_heads, 
                           key_dim=dec_key_dim, vocab_size=vocab_size, sentence_len=sentence_len, 
                           dff=dec_dff, dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` the context and x shall put in the
    # first argument.
    X, y = inputs # query=dec_input, context=enc_output
    X = self.encoder(X)  # (batch_size, strided len:2500/2, enco_dim)
    y = self.decoder(y, X)  # (batch_size, vocab_size, dec_dim)
    logits = self.final_layer(y)  # (batch_size, vocab_size, target_vocab_size)
    return logits


# In[training setting]
# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#   def __init__(self, d_model, warmup_steps=100):
#     super().__init__()
#     self.d_model = tf.cast(d_model, tf.float32)
#     self.warmup_steps = warmup_steps

#   def __call__(self, step):
#     step = tf.cast(step, dtype=tf.float32)
#     arg1 = tf.math.rsqrt(step)
#     arg2 = step * (self.warmup_steps ** -1.5)

#     return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# learning_rate = CustomSchedule(d_model=512)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# import matplotlib.pyplot as plt
# plt.plot(learning_rate(tf.range(300, dtype=tf.float32)))
# plt.ylabel('Learning Rate')
# plt.xlabel('Train Step')


def masked_loss(y_real, pred):
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  
  loss = loss_object(y_real, pred)
  mask = tf.logical_and(tf.not_equal(y_real, 41), tf.not_equal(y_real, 0))
  loss = tf.cast(loss, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(y_real, pred):
  pred = tf.argmax(pred, axis=2)
  # pred = tf.cast(pred, dtype=tf.int32)
  # y_real = tf.cast(y_real, dtype=tf.int32)
  match = y_real == pred
  mask = tf.logical_and(tf.not_equal(y_real, 41), tf.not_equal(y_real, 0))
  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)



checkpoint_path = '/Users/freya/Study/self-learning/Kaggle_brain_to_text/train/transformer.weights.h5'
transformer = Transformer(num_layers=1, enco_dim=512, enco_key_dim=128, num_heads=4, enc_dff=1000,
             dec_dim=88, dec_key_dim=22, vocab_size=43, sentence_len=149, dec_dff=400)


transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)


# reload model if the model exist
if os.path.exists(checkpoint_path):
    dummy_enc = tf.zeros((1, 2500, 512))
    dummy_dec = tf.zeros((1, 149))
    _ = transformer((dummy_enc, dummy_dec), training=False)
    transformer.load_weights(checkpoint_path)


# get dataset
batch_size = 64
root_dir = '/Users/freya/Study/self-learning/Kaggle_brain_to_text/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final/'
train_file_dir = glob.glob(os.path.join(root_dir, "**", "*_train.hdf5"), recursive=True)
train_dataset = generate_dataset(train_file_dir, batch_size, training=True)
valid_file_dir = glob.glob(os.path.join(root_dir, "**", "*_val.hdf5"), recursive=True)
val_dataset = generate_dataset(valid_file_dir, batch_size)


# transformer.run_eagerly = True
earlystop_callback = tf.keras.callbacks.EarlyStopping(patience=10, mode='min', 
                                            restore_best_weights=True, start_from_epoch=50)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min', # based on accuracy
    save_freq='epoch'
) # save model while model is better 


transformer.fit(train_dataset, epochs=300, validation_data=val_dataset, callbacks=[earlystop_callback, checkpoint_callback])
# transformer.save_weights('/Users/freya/Study/self-learning/Kaggle_brain_to_text/train/transformer.weights.h5')


# In[prediction and submission]
test_file_dir = glob.glob(os.path.join(root_dir, "**", "*_test.hdf5"), recursive=True)
test_dataset = generate_dataset(test_file_dir, batch_size, training=False)

import re
from nltk.corpus import cmudict
from collections import defaultdict
import kenlm
import csv


# pheno to text model
model = kenlm.Model('/Users/freya/Study/self-learning/3-gram.arpa') 

# vocab list 
LOGIT_TO_PHONEME = [
'BLANK',    # "BLANK" = CTC blank symbol
'AA', 'AE', 'AH', 'AO', 'AW',
'AY', 'B', 'CH', 'D', 'DH',
'EH', 'ER', 'EY', 'F', 'G',
'HH', 'IH', 'IY', 'JH', 'K',
'L', 'M', 'N', 'NG', 'OW',
'OY', 'P', 'R', 'S', 'SH',
'T', 'TH', 'UH', 'UW', 'V',
'W', 'Y', 'Z', 'ZH',
'|','<start>','<end>'    # "|" = silence token
]

vocab = np.array(LOGIT_TO_PHONEME)

# get text dictionary to convert phenomes to potencial text
cmu = cmudict.dict()
inv = defaultdict(set)

for word, pron_list in cmu.items():
    for pron in pron_list:
        cleaned = [re.sub(r"\d", "", p) for p in pron]
        key = " ".join(cleaned)
        inv[key].add(word.lower())

# trim the phenomes
def phoneme_seq_to_inv_key(phenomes):
    if isinstance(phenomes, np.ndarray):
        phenomes = phenomes.tolist()
    
    if "<end>" in phenomes: # delete idx after the first <end>
        idx = phenomes.index("<end>")
        phenomes = phenomes[:idx]
    
    key = " ".join(phenomes)
    return key


# get the potential text
def sentence_to_words_candidates(sentence, inv):
    words_phonemes = [w.strip() for w in sentence.split("|")]  # replace |
    candidates_list = []
    for wp in words_phonemes: 
        candidates = inv.get(wp, set())
        candidates_list.append(candidates)
    return candidates_list


def pheno_check(pheno):
    candidates = []
    for wp in pheno:
        if wp.upper() in model:
            candidates.append(wp.upper())
        elif wp in model:
            candidates.append(wp)
    return candidates


# get the best sentence candidate
def get_best_sentence_beam_search(candidates_list, model, beam_width=20):
    # get initial stats
    state = kenlm.State()
    model.BeginSentenceWrite(state)
    current_beams = [(0.0, state, [])]
    for candidates in candidates_list:
        candidates = pheno_check(candidates)

        if not candidates:
            candidates = {""} 
        
        next_beams = []
        for score, prev_state, history in current_beams:
            for word in candidates:
                
                new_state = kenlm.State()
                word_score = model.BaseScore(prev_state, word.upper(), new_state)
                
                new_total_score = score + word_score
                new_history = history + [word]
                
                next_beams.append((new_total_score, new_state, new_history))
        
        next_beams.sort(key=lambda x: x[0], reverse=True)
        current_beams = next_beams[:beam_width]
    
    if not current_beams:
        return None
        
    best_score, _, best_words = current_beams[0]
    return " ".join(best_words).strip()



def prediction_to_sentence(preds, vocab, inv):
    phonemes_idx = preds.numpy()
    phonemes = vocab[phonemes_idx]
    inv_keys = [phoneme_seq_to_inv_key(row) for row in phonemes]
    all_candidates = [sentence_to_words_candidates(s, inv) for s in inv_keys]

    all_best_sentences = []
    for i, candidates_list in enumerate(all_candidates):
        best_sentence = get_best_sentence_beam_search(candidates_list, model, beam_width=20)
        best_sentence = best_sentence.lower()# conver to lower letter 
        all_best_sentences.append(best_sentence) 
    return all_best_sentences



def greedy_decode(transformer, X, max_len=150,
                  start_token=41, end_token=42):
    """
    X: (batch_size, time_len, feature_dim)
    return: (batch_size, <= max_len)
    """

    batch_size = tf.shape(X)[0]
    enc_output = transformer.encoder(X)
    decoder_input = tf.fill([batch_size, 1], start_token) 

    for _ in range(max_len - 1):
        logits = transformer.final_layer(
            transformer.decoder(decoder_input, enc_output)
        )

        next_token_logits = logits[:, -1, :]

        next_token = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
        next_token = tf.expand_dims(next_token, axis=1)

        decoder_input = tf.concat([decoder_input, next_token], axis=1)

        if tf.reduce_all(tf.equal(next_token, end_token)):
            break
    return decoder_input


# runing loop
output = []
for batch, (inp, targ) in enumerate(test_dataset):
    final_preds = greedy_decode(transformer, inp)
    best_sentence = prediction_to_sentence(final_preds, vocab, inv)
    output.extend(best_sentence)
    

# save to csv file
filename = '/Users/freya/Study/self-learning/Kaggle_brain_to_text/submission.csv'

with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text'])
    
    for i, sentence in enumerate(output):
        writer.writerow([i, sentence])


