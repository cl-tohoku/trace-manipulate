#!/usr/bin/zsh

set -x
set -e

DATA_JSONL=$1 #jsonlを入れる
# OUT_DIR=$2 #最終的に学習等に使用するバイナリファイルの名前
MAX_LEN=$2
DATA_NO_JSONL=${DATA_JSONL%.jsonl}
OUT_DIR=data-bin/examples_n_features_${DATA_NO_JSONL#data/}_len${MAX_LEN}
python convert_synthetic_numeric_to_drop_with_type.py --data_jsonl ${DATA_JSONL}
./create_examples_n_features_with_type.sh ${DATA_NO_JSONL}_train_drop_format.json \
    ${DATA_NO_JSONL}_dev_drop_format.json ${OUT_DIR} ${MAX_LEN}
