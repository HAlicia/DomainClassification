#!/usr/bin/env bash

source activate mlp
# feature choice: ["tfidf", "binary", "count"], default: 'tfidf'
# num_words: N, most common N words to keep
# kernel choice: ["rbf", "linear", "poly", "sigmoid"] , default: 'rbf' (See sklearn doc)
echo "Start to run SVM ..."
python SVM_baseline.py --feat 'tfidf' --num_words 500 --kernel 'rbf' --max_iter -1
