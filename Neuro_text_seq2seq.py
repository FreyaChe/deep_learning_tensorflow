#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 16:34:06 2025

@author: freya
"""

import tensorflow as tf
import h5py
import glob
import os

# In[make data generator]
# read raw data 
def read_hdf5_file(file_path):
    # tensor to str
    file_path = file_path.decode('utf-8')
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
        
        X = tf.keras.utils.pad_sequences(inputs,maxlen=None, padding="post", dtype="float32") # padding data
        y_real = tf.keras.utils.pad_sequences(outputs,maxlen=None, padding="post") # padding data
    return X, y_real


# convert data to tensor formate
def tf_read_hdf5(file_path):
    inputs, outputs = tf.numpy_function(
        read_hdf5_file, 
        [file_path], 
        [tf.float32, tf.int32] 
    )
    outputs = tf.cast(outputs, tf.int64)
    return tf.data.Dataset.from_tensor_slices((inputs, outputs))


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
        
    # Padding Batch, pad data according to the longest data in one batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            [2500, 512],  # for X
            [150]         # for y_real
        ),
        padding_values=(0.0, tf.cast(0, tf.int64)) # padding with 0
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE) # preload data to accelerate speed 
    return dataset


# In[create seq2seq model]
# encoder 
class Encoder(tf.keras.Model):
    def __init__(self, batch_size, cov_units, gru_units, dropout=0.0):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.gru_units  = gru_units
        # conv1D
        self.Conv1D     = tf.keras.layers.Conv1D( filters=cov_units, kernel_size=6, 
                                                 strides=2, padding='same', activation='relu')
        # Gru
        self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=gru_units,
                                                                     activation='tanh',
                                                                     return_sequences=True,
                                                                     return_state=True,
                                                                     recurrent_initializer='glorot_uniform',
                                                                     dropout=dropout),
                                                 merge_mode='concat') 
    def call(self, x, training=True): # 
        # mask
        mask = tf.math.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        mask = mask[:, ::2]

        x = self.Conv1D(x)
        
        
        output_length = tf.shape(x)[1]
        mask = mask[:, :output_length]
        
        enc_outputs, forward_h, backward_h = self.gru(x, mask=mask, 
                                                 training=training
                                                 ) #dropout when training = true
        enc_states = tf.concat([forward_h, backward_h], axis=-1)
        return enc_outputs, enc_states, mask


# attention 
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, mask=None):
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        if mask is not None:
            # mask shape -> [batch, the time length of X, 1]
            mask = tf.cast(mask, dtype=score.dtype) 
            mask = tf.expand_dims(mask, axis=-1)
    
            # add a small value at 0
            score += (1.0 - mask) * -1e9
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
    

# decoder
class Decoder(tf.keras.Model):
    def __init__(self, batch_size, dec_units, vocab_size, embedding_dim, dropout=0.0): #  , batch_sz
        super(Decoder, self).__init__()
        # self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       activation = 'tanh',
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       dropout=dropout)
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(embedding_dim)  
        # context and x shall have same length so the units set as embedding_dim
        
    def call(self, x, dec_states, enc_output, mask=None, training=True):
        context, attention_score = self.attention( query=dec_states, values=enc_output, mask=mask)

        x = self.embedding(x) # (batch_size, 1, embedding_dim)

        # Concatenation: merge attention and input information
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        dec_output, dec_state = self.gru(x, initial_state=dec_states, training=training)
        # reshape output to 2D
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))
        predictions = self.dense(dec_output)   
        return predictions, dec_state, attention_score

    
# In[training setting]
# complie
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


# loss function, exclude padding result
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


# calculate accuracy
@tf.function
def calc_accuracy_tf(preds, targ):
    # preds: (batch, seq_len) -> prediction after argmax
    # targ: y_real
    
    # create a mask: ignore Start Token (41) and Padding (0)
    mask = tf.logical_and(tf.not_equal(targ, 41), tf.not_equal(targ, 0))
    
    # compare preds and y_real
    correct = tf.equal(preds, targ)
    
    # using Mask
    correct = tf.logical_and(correct, mask)
    
    mask_float = tf.cast(mask, dtype=tf.float32)
    correct_float = tf.cast(correct, dtype=tf.float32)
    accuracy = tf.reduce_sum(correct_float) / (tf.reduce_sum(mask_float) + 1e-9)
    return accuracy


# training step
@tf.function
def train_step(inp, targ, training=True):# , enc_hidden
    loss = 0
    pred_array = tf.TensorArray(tf.int64, size=targ.shape[1])

    with tf.GradientTape() as tape:
        # encoder
        enc_output, enc_states, mask = encoder(inp, training=training)
        dec_states = enc_states
        dec_input = tf.expand_dims(targ[:,0], 1)
        
        for t in range(1, targ.shape[1]):
            predictions, dec_states, _ = decoder(dec_input, dec_states, enc_output, mask=mask, training=training)
            # calculate loss in current t step
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t],1) # teach force, using real ou†put

            # calculate accuracy 
            pred_id = tf.argmax(predictions, axis=-1) # convert to argmax to calculate accuracy
            pred_array = pred_array.write(t, pred_id) 
            
    batch_loss = (loss / int(targ.shape[1]))
    final_preds = tf.transpose(pred_array.stack())
    batch_acc = calc_accuracy_tf(final_preds, targ)

    # update model
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss, batch_acc#, pred_sum


# valid step
@tf.function
def valid_step(inp, targ, training=False):  # , enc_hidden
    loss = 0
    pred_array = tf.TensorArray(tf.int64, size=targ.shape[1])

    # encoder
    enc_output, enc_states, mask = encoder(inp, training=training)
    dec_states = enc_states
    dec_input = tf.expand_dims(targ[:, 0], 1)

    for t in range(1, targ.shape[1]):
        predictions, dec_states, _ = decoder(
            dec_input, dec_states, enc_output, mask=mask, training=training)
        # calculate loss in current t step
        loss += loss_function(targ[:, t], predictions)
        # teach force, using real ou†put
        dec_input = tf.expand_dims(targ[:, t], 1)

        # calculate accuracy
        # convert to argmax to calculate accuracy
        pred_id = tf.argmax(predictions, axis=-1)
        pred_array = pred_array.write(t, pred_id)

    batch_loss = (loss / int(targ.shape[1]))
    final_preds = tf.transpose(pred_array.stack())
    batch_acc = calc_accuracy_tf(final_preds, targ)

    return batch_loss, batch_acc  # , pred_sum


# In[data and model setting]
# model setting 
batch_size, cov_units, gru_units = 64, 300, 172
dec_units, vocab_size, embedding_dim = 344, 43, 256

encoder = Encoder(batch_size, cov_units, gru_units) 
decoder = Decoder(batch_size, dec_units, vocab_size, embedding_dim)


# get dataset
root_dir = '/Users/freya/Study/self-learning/Kaggle_brain_to_text/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final/'
train_file_dir = glob.glob(os.path.join(root_dir, "**", "*_train.hdf5"), recursive=True)
train_dataset = generate_dataset(train_file_dir, batch_size, training=True)
valid_file_dir = glob.glob(os.path.join(root_dir, "**", "*_val.hdf5"), recursive=True)
val_dataset = generate_dataset(valid_file_dir, batch_size)


# save model with checkpoint
checkpoint_dir = '/Users/freya/Study/self-learning/Kaggle_brain_to_text/training_checkpoints_neu_tock'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# load model 
latest_ckpt = tf.train.latest_checkpoint('checkpoint_dir')
checkpoint.restore(latest_ckpt).expect_partial()


# In[training session]
EPOCHS = 200

# create acc and loss log(avoide interuption from print)
train_log_dir = '/Users/freya/Study/self-learning/Kaggle_brain_to_text/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# runing training
global_step = 0

for epoch in range(EPOCHS):
    # training session
    total_loss = 0
    accuracy = 0
    for batch, (inp, targ) in enumerate(train_dataset):
        batch_loss, batch_acc = train_step(inp, targ, training=True)
        accuracy += batch_acc
        total_loss += batch_loss
        if (batch + 1) % 10 == 0: 
            save_path = manager.save() # save model
            with train_summary_writer.as_default(): # batch result
                tf.summary.scalar('Batch/Loss', batch_loss, step=global_step)
                tf.summary.scalar('Batch/Accuracy', batch_acc, step=global_step)
            global_step += 1    
            
    with train_summary_writer.as_default(): # epoch result
        tf.summary.scalar('Epoch/Loss', total_loss/(batch+1), step=epoch)
        tf.summary.scalar('Epoch/Accuracy', accuracy/(batch+1), step=epoch)
    save_path = manager.save() # save model
    
    # validation session
    if (epoch + 1) % 5 == 0:
        total_loss = 0
        accuracy = 0
        for batch, (inp, targ) in enumerate(val_dataset):
            batch_loss, preds = valid_step(inp, targ, training=False)
            accuracy += batch_acc
            total_loss += batch_loss
        with train_summary_writer.as_default():
                tf.summary.scalar('Valid/Loss', total_loss/(batch+1), step=epoch)
                tf.summary.scalar('Valid/Accuracy', accuracy/(batch+1), step=epoch)


# In[prediction and submission]
test_file_dir = glob.glob(os.path.join(root_dir, "**", "*_test.hdf5"), recursive=True)
test_dataset = generate_dataset(valid_file_dir, batch_size, training=False)

from pyctcdecode import build_ctcdecoder
import numpy as np
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
        phonemes = phenomes.tolist()
    
    if "<end>" in phenomes: # delete idx after the first <end>
        idx = phenomes.index("<end>")
        phonemes = phenomes[:idx]
    
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


# get the best sentence candidate
def get_best_sentence_beam_search(candidates_list, model, beam_width=10):
    # get initial stats
    state = kenlm.State()
    model.BeginSentenceWrite(state)
    current_beams = [(0.0, state, [])]

    for candidates in candidates_list:
        valid_candidates = set()
        for w in candidates:
            if w and w.strip():
                valid_candidates.add(w.upper())
        
        if not valid_candidates:
            valid_candidates = {"<unk>"}
        
        next_beams = []
        
        for score, prev_state, history in current_beams:
            for word in candidates:
                new_state = kenlm.State()
                word_score = model.BaseScore(prev_state, word, new_state)
                
                new_total_score = score + word_score
                new_history = history + [word]
                
                next_beams.append((new_total_score, new_state, new_history))
        
        next_beams.sort(key=lambda x: x[0], reverse=True)
        current_beams = next_beams[:beam_width]

    if not current_beams:
        return None
        
    best_score, _, best_words = current_beams[0]
    return " ".join(best_words).strip() # joint sentence and remove space 



# prediction step
@tf.function
def pred_step(inp, batch_size):
    pred_array = tf.TensorArray(tf.int64, size=targ.shape[1]-1)

    # encoder
    enc_output, enc_states, mask = encoder(inp, training=False)
    dec_states = enc_states
    dec_input = tf.expand_dims([41] * batch_size, 1)

    for t in range(1, targ.shape[1]):
        predictions, dec_states, _ = decoder(
            dec_input, dec_states, enc_output, mask=mask, training=False)
        
        pred_id = tf.argmax(predictions, axis=-1)
        dec_input = tf.expand_dims(pred_id, 1)
        pred_array = pred_array.write(t-1, pred_id)

    final_preds = tf.transpose(pred_array.stack())

    return final_preds


def prediction_to_sentence(preds, vocab, inv):
    phonemes_idx = final_preds.numpy()
    phonemes = vocab[phonemes_idx]
    inv_keys = [phoneme_seq_to_inv_key(row) for row in phonemes]
    all_candidates = [sentence_to_words_candidates(s, inv) for s in inv_keys]

    all_best_sentences = []
    for i, candidates_list in enumerate(all_candidates):
        lens = [len(c) for c in candidates_list]

        best_sentence = get_best_sentence_beam_search(candidates_list, model, beam_width=20)
        all_best_sentences.append(best_sentence.lower()) # conver to lower letter 
    return all_best_sentences


# runing loop
output = []
for batch, (inp, targ) in enumerate(pred_step):
    final_preds = pred_step(inp, batch_size)
    best_sentence = prediction_to_sentence(final_preds, vocab, inv)
    output.extend(batch_sentences)
 

# save to csv file
filename = '/Users/freya/Study/self-learning/Kaggle_brain_to_text/submission.csv'

with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text'])
    
    for i, sentence in enumerate(output):
        writer.writerow([i, sentence])
