#!/bin/bash

triplet_slice=("98-40" "98-80" "90-70" "80-60" "70-50" "60-40")
for i in ${triplet_slice[@]}; do
python matchmaker/train.py --config-file config/train/bert-dot.yaml config/train/data/trec-2019-dl-dataset.yaml --run-name bert-dot_tfidf_${i} \
--config-overwrites "train_tsv: /data/tfink/kodicare/trec-2019-dl/doc_ret/triplets_tfidf_${i}.train.txt"
model=(/data/tfink/kodicare/trec-2019-dl/doc_ret/experiments/*_bert-dot_tfidf_${i})
python matchmaker/dense_retrieval.py encode+index+search --run-name bert-dot_tfidf_${i}_eval --config config/dense_retrieval/trec-2019-dl.yaml \
--config-overwrites "trained_model: ${model[@]}"
continue_folder=(/data/tfink/kodicare/trec-2019-dl/doc_ret/experiments/*_bert-dot_tfidf_${i}_eval)
python matchmaker/dense_retrieval.py index+search --run-name bert-dot_tfidf_${i}_eval2 --config config/dense_retrieval/trec-2020-dl.yaml \
--config-overwrites "trained_model: ${model[@]}, continue_folder: ${continue_folder[@]}"
done
