#!/usr/bin/zsh

set -x
set -e

DROP_TRAIN=$1
DROP_EVAL=$2
OUT_DIR=$3
MAX_LEN=$4

python create_examples_n_features_with_type.py --split train --drop_json ${DROP_TRAIN} --output_dir ${OUT_DIR}  --max_seq_length ${MAX_LEN} --max_decoding_steps 11 --max_n_samples -1

python create_examples_n_features_with_type.py --split eval --drop_json ${DROP_EVAL} --output_dir ${OUT_DIR}  --max_seq_length ${MAX_LEN} --max_decoding_steps 11
