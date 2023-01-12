"""A Transformer with a BERT encoder and BERT decoder with extensive weight tying."""
# In each decoder layer, the self attention params are also used for source attention, 
# thereby allowing us to use BERT as a decoder as well.
# Most of the code is taken from HuggingFace's repo.

from __future__ import absolute_import

import argparse
import logging
import os, sys, random, jsonlines, shutil, time
import ujson as json
from scipy.special import softmax
from io import open
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm, trange

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from create_examples_n_features_with_type import DropExample, DropFeatures, read_file, write_file, split_digits


START_TOK, END_TOK, SPAN_SEP, IGNORE_IDX, MAX_SPANS = '@', '\\', ';', 0, 6



ModelFeatures = namedtuple("ModelFeatures", "example_id input_ids input_mask segment_ids label_ids head_type q_spans p_spans")
class DropDataset(TensorDataset):
    def __init__(self, args, split='train'):
        logging.info(f"Loading {split} examples and features.")
        direc = args.examples_n_features_dir
        if split == 'train':
            examples = read_file(direc + '/train_examples.pkl')
            drop_features = read_file(direc + '/train_features.pkl')
        else:
            examples = read_file(direc + '/eval_examples.pkl')
            drop_features = read_file(direc + '/eval_features.pkl')
        
        num_samples = len(examples)
        self.max_dec_steps = len(drop_features[0].decoder_label_ids)
        
        features = []
        for i, (example, drop_feature) in enumerate(zip(examples, drop_features)):
            features.append(self.convert_to_input_features(example, drop_feature))
            if split == 'train' and args.num_train_samples >= 0 and len(features) >= args.num_train_samples:
                break
        print()
#         assert i == num_samples - 1
        self.num_samples = len(features)
        self.seq_len = drop_features[0].max_seq_length
        self.examples = examples
        self.numbers = torch.unsqueeze(torch.tensor([float(e.answer_texts[0].replace(" ","")) for e in examples]),1)
        self.drop_features = drop_features
        self.features = features
        self.example_ids = [f.example_id for f in features]
        self.input_ids = torch.tensor([f.input_ids for f in features]).long()
        self.input_mask = torch.tensor([f.input_mask for f in features]).long()
        self.segment_ids = torch.tensor([f.segment_ids for f in features]).long()
        self.label_ids = torch.tensor([f.label_ids for f in features]).long()
        self.head_type = torch.tensor([f.head_type for f in features]).long()
        self.q_spans = torch.tensor([f.q_spans for f in features]).long()
        self.p_spans = torch.tensor([f.p_spans for f in features]).long()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (self.input_ids[item], self.input_mask[item], self.segment_ids[item], self.label_ids[item], 
                self.head_type[item], self.q_spans[item], self.p_spans[item],self.numbers[item])
    
    def convert_to_input_features(self, drop_example, drop_feature):
        max_seq_len = drop_feature.max_seq_length
        
        # input ids are padded by 0
        input_ids = drop_feature.input_ids
        input_ids += [IGNORE_IDX] * (max_seq_len - len(input_ids))
        
        # input mask is padded by 0
        input_mask = drop_feature.input_mask
        input_mask += [0] * (max_seq_len - len(input_mask))
        
        # segment ids are padded by 0
        segment_ids = drop_feature.segment_ids
        segment_ids += [0] * (max_seq_len - len(segment_ids))
        
        # we assume dec label ids are already padded by 0s
        decoder_label_ids = drop_feature.decoder_label_ids
        assert len(decoder_label_ids) == self.max_dec_steps 
        #decoder_label_ids += [0] * (MAX_DECODING_STEPS - len(decoder_label_ids))
        
        # for span extraction head, ignore idx == -1
        question_len = segment_ids.index(1) if 1 in segment_ids else len(segment_ids)
        starts, ends = drop_feature.start_indices, drop_feature.end_indices
        q_spans, p_spans = [], []
        for st, en in zip(starts, ends):
            if any([x < 0 or x >= max_seq_len for x in [st, en]]):
                continue
            elif all([x < question_len for x in [st, en]]):
                q_spans.append([st, en])
            elif all([question_len <= x for x in [st, en]]):
                p_spans.append([st, en])
        q_spans, p_spans = q_spans[:MAX_SPANS], p_spans[:MAX_SPANS]
        head_type = 1 if q_spans or p_spans else -1
        q_spans += [[-1,-1]]*(MAX_SPANS - len(q_spans))
        p_spans += [[-1,-1]]*(MAX_SPANS - len(p_spans))
                
        return ModelFeatures(drop_feature.example_index, input_ids, input_mask, segment_ids,
                             decoder_label_ids, head_type, q_spans, p_spans)

    
