#!/bin/bash


#CUDA_VISIBLE_DEVICES=0,1,2,3 python matchmaker/train.py --config-file config/train/bert-dot-dual.yaml config/train/data/trec-2019-dl-dataset.yaml --run-name bert-dot-dual_bert_test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python matchmaker/dense_retrieval.py encode+index+search --run-name bert-dot_random-baseline_eval --config config/dense_retrieval/trec-2019-dl.yaml

triplet_slice=("98-50" "90-70" "90-60" "90-50" "80-60" "80-50" "70-50")
for i in ${triplet_slice[@]}; do
python matchmaker/train.py --config-file config/train/bert-dot.yaml config/train/data/trec-2019-dl-dataset.yaml --run-name bert-dot_bert_${i} \
--config-overwrites "train_tsv: /data/tfink/kodicare/trec-2019-dl/doc_ret/triplets_bert_${i}.train.txt"
model=(/data/tfink/kodicare/trec-2019-dl/doc_ret/experiments/*_bert-dot_bert_${i})
python matchmaker/dense_retrieval.py encode+index+search --run-name bert-dot_bert_${i}_eval --config config/dense_retrieval/trec-2019-dl.yaml \
--config-overwrites "trained_model: ${model[@]}"
continue_folder=(/data/tfink/kodicare/trec-2019-dl/doc_ret/experiments/*_bert-dot_bert_${i}_eval)
python matchmaker/dense_retrieval.py index+search --run-name bert-dot_bert_${i}_eval2 --config config/dense_retrieval/trec-2020-dl.yaml \
--config-overwrites "trained_model: ${model[@]}, continue_folder: ${continue_folder[@]}"
done