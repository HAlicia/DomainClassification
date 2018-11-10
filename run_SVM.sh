#!/usr/bin/env bash

source activate jd
# feature choice: ["tfidf", "binary", "count"], default: 'tfidf'
# num_words: N, most common N words to keep
# kernel choice: ["rbf", "linear", "poly", "sigmoid"] , default: 'rbf' (See sklearn doc)
python --feat 'tfidf' --num_words 5000 --kernel 'rbf' --max_iter -1
