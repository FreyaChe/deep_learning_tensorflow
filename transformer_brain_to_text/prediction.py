#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 10:35:52 2026

@author: freya
"""


import re
import kenlm
import numpy as np
import tensorflow as tf
from dataset import get_dataset
from nltk.corpus import cmudict
from collections import defaultdict
from config import batch_size, gram_path, csv_path, weight_path, cfg_model
from transformer_model import generate_transformer
import pandas as pd


# pheno to text model
text_model = kenlm.Model(str(gram_path)) 

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
        if wp.upper() in text_model:
            candidates.append(wp.upper())
        elif wp in text_model:
            candidates.append(wp)
    return candidates


# get the best sentence candidate
def get_best_sentence_beam_search(candidates_list, text_model, beam_width=150):
    # get initial stats
    state = kenlm.State()
    text_model.BeginSentenceWrite(state)
    current_beams = [(0.0, state, [])]
    for candidates in candidates_list:
        candidates = pheno_check(candidates)

        if not candidates:
            candidates = {""} 
        
        next_beams = []
        for score, prev_state, history in current_beams:
            for word in candidates:
                
                new_state = kenlm.State()
                word_score = text_model.BaseScore(prev_state, word.upper(), new_state)
                
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
    phonemes_idx = phonemes_idx[:,1:]
    phonemes = vocab[phonemes_idx]
    inv_keys = [phoneme_seq_to_inv_key(row) for row in phonemes]
    all_candidates = [sentence_to_words_candidates(s, inv) for s in inv_keys]

    all_best_sentences = []
    for i, candidates_list in enumerate(all_candidates):
        best_sentence = get_best_sentence_beam_search(candidates_list, text_model, beam_width=20)
        best_sentence = best_sentence.lower()# conver to lower letter 
        all_best_sentences.append(best_sentence) 
    return all_best_sentences


def beam_search_decode(model, X, max_len=150, start_token=41, end_token=42,
                       beam_width=4, length_penalty=0.7):
    """
    Beam search decoding for transformer model.
    
    Args:
        model: Transformer model
        X: encoder input, shape (batch_size, seq_len, channels)
        max_len: maximum output sequence length
        start_token: start token id (41)
        end_token: end token id (42)
        beam_width: number of beams (recommended 4~8)
        length_penalty: length normalization exponent (0.6~0.8)
    
    Returns:
        best_sequences: shape (batch_size, seq_len)
    """
    batch_size = tf.shape(X)[0].numpy()
    enc_output, enc_mask = model.encoder(X, training=False)

    # beams[batch_idx] = list of (sequence, cumulative_log_prob)
    # sequence shape: (1, current_len)
    beams = []
    for b in range(batch_size):
        initial_seq = tf.fill([1, 1], tf.cast(start_token, tf.int64))
        beams.append([(initial_seq, 0.0)])

    for step in range(max_len - 1):
        new_beams = [[] for _ in range(batch_size)]

        for batch_idx in range(batch_size):
            for seq, cum_score in beams[batch_idx]:
                
                # if last token is end_token, keep beam as-is
                if seq.shape[1] > 1 and seq[0, -1].numpy() == end_token:
                    new_beams[batch_idx].append((seq, cum_score))
                    continue

                # decoder forward pass
                dec_output, _ = model.decoder(
                    seq,
                    enc_output[batch_idx:batch_idx + 1],
                    enc_mask[batch_idx:batch_idx + 1]
                )
                logits = model.final_layer(dec_output)
                next_token_logits = logits[0, -1, :]  # (vocab_size,)

                # use log_softmax for numerically stable scores
                log_probs = tf.nn.log_softmax(next_token_logits)
                top_log_probs, top_indices = tf.nn.top_k(log_probs, k=beam_width)

                for i in range(beam_width):
                    next_token = tf.cast(top_indices[i], tf.int64)
                    token_log_prob = top_log_probs[i].numpy()

                    new_seq = tf.concat(
                        [seq, tf.reshape(next_token, [1, 1])],
                        axis=1
                    )
                    new_score = cum_score + token_log_prob
                    new_beams[batch_idx].append((new_seq, new_score))

        # keep top beam_width beams with length normalization
        for batch_idx in range(batch_size):
            new_beams[batch_idx].sort(
                key=lambda x: x[1] / (x[0].shape[1] ** length_penalty),
                reverse=True
            )
            beams[batch_idx] = new_beams[batch_idx][:beam_width]

        # early stop if all beams ended
        all_ended = all(
            seq[0, -1].numpy() == end_token
            for batch_idx in range(batch_size)
            for seq, _ in beams[batch_idx]
        )
        if all_ended:
            break

    # pick best beam per sample (highest length-normalized score)
    best_sequences = []
    for batch_idx in range(batch_size):
        best_seq, _ = max(
            beams[batch_idx],
            key=lambda x: x[1] / (x[0].shape[1] ** length_penalty)
        )
        best_sequences.append(best_seq[0])  # shape: (seq_len,)

    # pad to same length
    max_seq_len = max(seq.shape[0] for seq in best_sequences)
    padded = []
    for seq in best_sequences:
        pad_len = max_seq_len - seq.shape[0]
        if pad_len > 0:
            padding = tf.zeros([pad_len], dtype=tf.int64)
            seq = tf.concat([seq, padding], axis=0)
        padded.append(seq)

    return tf.stack(padded, axis=0)  # (batch_size, max_seq_len)



# def greedy_decode(model, X, max_len=150,
#                   start_token=41, end_token=42):
#     """
#     X: (batch_size, time_len, feature_dim)
#     return: (batch_size, <= max_len)
#     """

#     batch_size = tf.shape(X)[0]
#     enc_output, enc_mask = model.encoder(X)
#     decoder_input = tf.fill([batch_size, 1], start_token) 

#     for _ in range(max_len - 1):
#         dec_output, _ = model.decoder(decoder_input, enc_output, enc_mask)
#         logits = model.final_layer(dec_output)
        
#         # next_token_logits = logits[:, -1, :]

#         next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
#         # next_token = tf.expand_dims(next_token, axis=1)

#         decoder_input = tf.concat([decoder_input, next_token], axis=1)

#         if tf.reduce_all(tf.equal(next_token, end_token)):
#             break
#     return decoder_input



# test_dataset = get_dataset('test', batch_size)
# transformer = generate_transformer(cfg_model, weight_path)

# # runing loop
# output = []
# for batch, (inp, targ) in enumerate(test_dataset.take(1)):
#     X, targ = inp
#     final_preds = beam_search_decode_top2(transformer, X)
#     best_sentence = prediction_to_sentence(final_preds, vocab, inv)
#     output.extend(best_sentence)


valid_dataset = get_dataset('valid', batch_size)
transformer = generate_transformer(cfg_model, weight_path)


output = []
y_real = []
# runing loop
for batch, (inp, y) in enumerate(valid_dataset):
    X, targ = inp
    final_preds = beam_search_decode(transformer, X, beam_width=4)
    best_sentence = prediction_to_sentence(final_preds, vocab, inv)
    output.extend(best_sentence)
    
    orig_sentence = prediction_to_sentence(y, vocab, inv)
    y_real.extend(orig_sentence)


# save to csv file
df = pd.DataFrame({
    "real_sentence": y_real,
    "prediction": output
})

df.to_csv(csv_path, index=False)

