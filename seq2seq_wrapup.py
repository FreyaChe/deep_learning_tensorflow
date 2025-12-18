# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 16:55:07 2025
@author: freya

brain to text with seq2seq
"""

import tensorflow as tf
import h5py
import glob
import os


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
    return tf.data.Dataset.from_tensor_slices(((X, y), y))


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



# In[create seq2seq model]
# encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, cov_units, gru_units, dropout=0.1):
        super(Encoder, self).__init__()
        self.gru_units  = gru_units
        # conv1D
        self.Conv1D     = tf.keras.layers.Conv1D( filters=cov_units, kernel_size=6,
                                                 strides=2, padding='same', activation='relu')
        # bidirectional Gru
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

        x = self.Conv1D(x) # shape(batch, strided len:2500/2, channels:512)


        output_length = tf.shape(x)[1]
        mask = mask[:, :output_length] # present as True/False
        enc_outputs, forward_h, backward_h = self.gru(x, mask=mask)
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
            mask = tf.cast(mask, dtype=score.dtype) # convert mask to 1/0
            mask = tf.expand_dims(mask, axis=-1)

            # add a small value at 0
            score += (1.0 - mask) * -1e9
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, dec_units, vocab_size, embedding_dim, dropout=0.1):
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

        x = self.embedding(x) # shape(batch_size, 1, embedding_dim)

        # Concatenation: merge attention and input information
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        dec_output, dec_state = self.gru(x, initial_state=dec_states)
        # reshape output to 2D
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))
        predictions = self.dense(dec_output)
        return predictions, dec_state, attention_score


# Model
class Model_seq2seq(tf.keras.Model):
    def __init__(self, *, cov_units, gru_units, dec_units, vocab_size, embedding_dim):
      super().__init__()
      self.encoder = Encoder(cov_units, gru_units)
      self.decoder = Decoder(dec_units, vocab_size, embedding_dim)

    def call(self, inputs):
      X, y_real = inputs
      enc_output, enc_states, mask = self.encoder(X)
      dec_states = enc_states
      dec_input = tf.expand_dims(y_real[:,0], 1)
      logits = tf.TensorArray(tf.float32, size=y_real.shape[1])
      for t in range(1, y_real.shape[1]):
        predictions, dec_states, _ = self.decoder(dec_input, dec_states, enc_output, mask=mask)
        dec_input = tf.expand_dims(y_real[:, t],1) # teach force
        logits = logits.write(t, predictions)
      logits = tf.transpose(logits.stack(), [1, 0, 2])
      return logits


# In[training setting]
# complie
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def masked_loss(y_real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(y_real, pred)
    mask = tf.logical_and(tf.not_equal(y_real, 41), tf.not_equal(y_real, 0))
    loss = tf.cast(loss, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    loss *= mask
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_accuracy(y_real, pred):
    pred = tf.argmax(pred, axis=-1)
    match_y = tf.equal(y_real, pred)

    mask = tf.logical_and(tf.not_equal(y_real, 41), tf.not_equal(y_real, 0))
    correct = tf.logical_and(match_y, mask)

    mask = tf.cast(mask, dtype=tf.float32)
    correct = tf.cast(correct, dtype=tf.float32)
    return tf.reduce_sum(correct)/tf.reduce_sum(mask)


cov_units, gru, dec_units, vocab_size, embedding_dim = 300, 172, 344, 43, 256
batch_size = 12

model_seq = Model_seq2seq(cov_units=cov_units, gru_units=gru, dec_units=dec_units,
              vocab_size=vocab_size, embedding_dim=embedding_dim)

model_seq.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])


# In[training dataset]
root_dir = '/Users/freya/Study/self-learning/Kaggle_brain_to_text/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final/'
train_file_dir = glob.glob(os.path.join(root_dir, "**", "*_train.hdf5"), recursive=True)
train_dataset = generate_dataset(train_file_dir, batch_size, training=True)
valid_file_dir = glob.glob(os.path.join(root_dir, "**", "*_val.hdf5"), recursive=True)
val_dataset = generate_dataset(valid_file_dir, batch_size)


early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, mode='min', 
                                            restore_best_weights=True, start_from_epoch=50)

checkpoint_path = '/Users/freya/Study/self-learning/Kaggle_brain_to_text/train/seq2seq.weights.h5'

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True, # only save weight
    save_best_only=True,
    monitor='val_loss',
    mode='min', # based on accuracy
    save_freq='epoch'
) # save model while model is better 


model_seq.fit(train_dataset, validation_data=val_dataset, epochs=300, callbacks=[early_stopping, checkpoint])


# In[prediction and submission]
test_file_dir = glob.glob(os.path.join(root_dir, "**", "*_test.hdf5"), recursive=True)
test_dataset = generate_dataset(test_file_dir, batch_size, training=False)

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
    # phonemes_idx = preds#.numpy()
    phonemes = vocab[preds]
    inv_keys = [phoneme_seq_to_inv_key(row) for row in phonemes]
    all_candidates = [sentence_to_words_candidates(s, inv) for s in inv_keys]

    all_best_sentences = []
    for i, candidates_list in enumerate(all_candidates):
        best_sentence = get_best_sentence_beam_search(candidates_list, model, beam_width=20)
        best_sentence = best_sentence.lower()# conver to lower letter
        all_best_sentences.append(best_sentence)
    return all_best_sentences


def predict(self, inputs, max_length=150, end_token=42):
    X, _ = inputs
    enc_output, enc_states, mask = self.encoder(X)
    dec_states = enc_states
    dec_input = tf.expand_dims([41] * batch_size, 1)

    result = []

    for t in range(max_length):
        predictions, dec_states, _ = self.decoder(dec_input, dec_states, enc_output, mask=mask)
        predicted_id = tf.argmax(predictions, axis=-1).numpy()
        result.append(predicted_id)
        if tf.reduce_all(tf.equal(predicted_id, end_token)):
            result = np.asarray(result)
            break
    result = np.asarray(result)
    return result


# runing loop
output = []
for batch, (inp, targ) in enumerate(test_dataset):
    final_preds = predict(model_seq, inp)
    best_sentence = prediction_to_sentence(final_preds, vocab, inv)
    output.extend(best_sentence)


# save to csv file
filename = '/Users/freya/Study/self-learning/Kaggle_brain_to_text/submission.csv'

with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text'])

    for i, sentence in enumerate(output):
        writer.writerow([i, sentence])