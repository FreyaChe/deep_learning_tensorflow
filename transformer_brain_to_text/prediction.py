#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 10:35:52 2026

@author: freya
"""


import re
import kenlm
import csv
import numpy as np
import tensorflow as tf
from dataset import get_dataset
from nltk.corpus import cmudict
from collections import defaultdict
from config import batch_size, gram_path, csv_path, weight_path, cfg_model
from transformer_model import generate_transformer


# pheno to text model
text_model = kenlm.Model(gram_path) 

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
def get_best_sentence_beam_search(candidates_list, text_model, beam_width=20):
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


def beam_search_decode_top2(model, X, max_len=150,
                            start_token=41, end_token=42, beam_width=2):

    batch_size = tf.shape(X)[0]
    enc_output, enc_mask = model.encoder(X)
    
    # beams[i] = list of (sequence, cumulative_score)
    # initial beams 
    beams = []
    for b in range(batch_size):
        initial_seq = tf.fill([1, 1], start_token)
        beams.append([(initial_seq, 0.0)]) # beams includes: sequence(inital=start_token), start score(initial=0)
        
    
    for step in range(max_len - 1):
        new_beams = [[] for _ in range(batch_size)]
        
        for batch_idx in range(batch_size): # calculates 
            for seq, cum_score in beams[batch_idx]:
                if seq.shape[1] > 1 and seq[0, -1].numpy() == end_token:
                    new_beams[batch_idx].append((seq, cum_score))
                    continue
                
                dec_output, _ = model.decoder(seq, enc_output[batch_idx:batch_idx+1], enc_mask[batch_idx:batch_idx+1])
                logits = model.final_layer(dec_output)
                next_token_logits = logits[0, -1, :]  # (vocab_size,)
                
                top2_logits, top2_indices = tf.nn.top_k(next_token_logits, k=beam_width)
                
                for i in range(beam_width):
                    next_token = top2_indices[i]
                    token_score = top2_logits[i].numpy()
                    
                    new_seq = tf.concat([seq, tf.expand_dims(tf.expand_dims(next_token, 0), 0)], axis=1)
                    new_score = cum_score + token_score
                    new_beams[batch_idx].append((new_seq, new_score))
        
        for batch_idx in range(batch_size):
            new_beams[batch_idx].sort(key=lambda x: x[1], reverse=True)
            beams[batch_idx] = new_beams[batch_idx][:beam_width]
        
        all_ended = True
        for batch_idx in range(batch_size):
            for seq, _ in beams[batch_idx]:
                if seq[0, -1].numpy() != end_token:
                    all_ended = False
                    break
            if not all_ended:
                break
        
        if all_ended:
            break
    
    best_sequences = []
    for batch_idx in range(batch_size):
        best_seq, _ = beams[batch_idx][0]
        best_sequences.append(best_seq[0])
    
    max_seq_len = max(seq.shape[0] for seq in best_sequences)
    padded_sequences = []
    for seq in best_sequences:
        if seq.shape[0] < max_seq_len:
            padding = tf.fill([max_seq_len - seq.shape[0]], end_token)
            seq = tf.concat([seq, padding], axis=0)
        padded_sequences.append(seq)
    
    return tf.stack(padded_sequences, axis=0)




test_dataset = get_dataset('test', batch_size)
transformer = generate_transformer(cfg_model, weight_path)

# runing loop
output = []
for batch, (inp, targ) in enumerate(test_dataset):
    # for batch, (inp, targ) in enumerate(train_dataset):
    X, targ = inp
    final_preds = beam_search_decode_top2(transformer, X)
    best_sentence = prediction_to_sentence(final_preds, vocab, inv)
    output.extend(best_sentence)


# save to csv file
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text'])
    
    for i, sentence in enumerate(output):
        writer.writerow([i, sentence])

