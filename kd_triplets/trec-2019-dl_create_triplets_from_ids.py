import os
import json
import argparse
from tqdm import tqdm
import time
import random
import hydra
from omegaconf import DictConfig


# Create Triplets for dense retrieval training in the format: query-text<tab>pos-text<tab>neg-text.


def get_queries(queries_file):
    queries = {}
    with open(queries_file, "r") as fp:
        for queries_line in fp:
            q_id, q_text = queries_line.strip().split(sep="\t")
            queries[q_id] = q_text
    return queries


def get_documents(document_file, n_docs):
    documents = {}
    with open(document_file, "r") as fp:
        for doc_line in tqdm(fp, total=n_docs, desc="Load Documents"):
            data = doc_line.strip().split(sep="\t")
            if len(data) != 4:
                #print(len(data), data)
                continue
            doc_id, doc_url, doc_title, doc_text = data
            documents[doc_id] = doc_text
    return documents


def get_document_ids(document_file):
    doc_ids = []
    doc_ids_inv = {}
    doc_ids_int64 = []
    with open(document_file, "r") as fp:
        for doc_line in fp:
            data = doc_line.strip().split(sep="\t")
            if len(data) != 4:
                #print(len(data), data)
                continue
            doc_id, doc_url, doc_title, doc_text = data
            doc_ids_inv[doc_id] = len(doc_ids)
            doc_ids_int64.append(len(doc_ids))
            doc_ids.append(doc_id)
    return doc_ids, doc_ids_inv, doc_ids_int64


def read_qrels(qrels_file):
    relevance_judgements = {}
    with open(qrels_file, "r") as fp:
        for qrel_line in fp:
            q_id, _, doc_id, relevance = qrel_line.strip().split()
            relevance = int(relevance)
            if relevance == 1:
                if q_id not in relevance_judgements:
                    relevance_judgements[q_id] = set()
                relevance_judgements[q_id].add(doc_id)
    return relevance_judgements


def relevance_judgement_iter(relevance_judgements, doc_ids_inv):
    for q_id, relevant_doc_ids in tqdm(relevance_judgements.items()):
        for pos_doc_id in relevant_doc_ids:
            if pos_doc_id not in doc_ids_inv:
                continue
            yield q_id, pos_doc_id, relevant_doc_ids


def create_triplets(cfg, triplet_slice, documents, queries, upper_bound, lower_bound):
    random_seed = cfg.random_seed
    triplet_folder = cfg.triplet_folder
    document_file = cfg.document_file
    qrels_file = cfg.qrels_file
    kd_experiment_name = cfg.kd_experiment_name
    triplet_id_file = os.path.join(triplet_folder, f"triplets_{kd_experiment_name}.train.id")
    truncate = cfg.truncate
    analysis = cfg.analysis
    output_seq_length = cfg.output_seq_length
    # total number of negatives per query
    query_total_negatives = cfg.total_negatives
    # hard negatives must be <= total_negatives
    query_hard_negatives = cfg.hard_negatives

    triplet_slice_name = f"triplets_{kd_experiment_name}_{triplet_slice}.train.txt"
    print(f"Processing Triplets {triplet_slice_name}")

    doc_ids, doc_ids_inv, doc_ids_int64 = get_document_ids(document_file)
    relevance_judgements = read_qrels(qrels_file)

    random.seed(random_seed)

    print("Loading Hard Negatives")
    triplets_hard_negatives = {}
    with open(triplet_id_file, 'r') as fp:
        for line in fp:
            q_id, pos_doc_id, neg_doc_id, s = line.strip().split()
            s = float(s)
            if s >= upper_bound or s < lower_bound:
                continue
            if (q_id, pos_doc_id) not in triplets_hard_negatives:
                triplets_hard_negatives[(q_id, pos_doc_id)] = []
            triplets_hard_negatives[(q_id, pos_doc_id)].append(neg_doc_id)

    print(f"Creating Triplets")
    triplets = []
    for q_id, pos_doc_id in tqdm(triplets_hard_negatives.keys()):
        relevant_doc_ids = relevance_judgements[q_id]
        hard_negative_ids = triplets_hard_negatives[(q_id, pos_doc_id)]
        neg_doc_ids = random.sample(doc_ids, k=query_total_negatives)
        if query_hard_negatives > len(hard_negative_ids):
            k = len(hard_negative_ids)
        else:
            k = query_hard_negatives
        sample_hard_neg_doc_ids = random.sample(hard_negative_ids, k=k)
        # replace the first few random negatives with the hard negatives sample
        for i in range(len(sample_hard_neg_doc_ids)):
            neg_doc_ids[i] = sample_hard_neg_doc_ids[i]
        for neg_doc_id in neg_doc_ids:
            if neg_doc_id in relevant_doc_ids:
                continue
            triplets.append((q_id, pos_doc_id, neg_doc_id))

    print(f"Shuffling Triplets")
    # shuffle training data
    random.shuffle(triplets)

    print(f"Writing Triplets")
    # now with buffering
    with open(os.path.join(triplet_folder, triplet_slice_name), 'w') as fp_out:
        buffer = []
        for q_id, pos_doc_id, neg_doc_id in tqdm(triplets):
            pos_doc = documents[pos_doc_id]
            neg_doc = documents[neg_doc_id]
            if truncate:
                pos_doc = " ".join(pos_doc.split()[:output_seq_length])
                neg_doc = " ".join(neg_doc.split()[:output_seq_length])
            if analysis:
                buffer.append(f"{queries[q_id]}\n===\n{pos_doc}\n===\n{neg_doc}\t{s}\n\n\n")
            else:
                buffer.append(f"{queries[q_id]}\t{pos_doc}\t{neg_doc}\n")
            if len(buffer) >= 1000:
                fp_out.write("".join(buffer))
                buffer = []
        if len(buffer) > 0:
            fp_out.write("".join(buffer))


@hydra.main(version_base=None, config_path=".", config_name=None)
def main(cfg: DictConfig):
    triplet_files = cfg.triplet_files
    n_docs = cfg.n_docs
    document_file = cfg.document_file
    queries_file = cfg.queries_file

    documents = get_documents(document_file, n_docs=n_docs)
    queries = get_queries(queries_file)

    for triplet_slice, kwargs in triplet_files.items():

        create_triplets(cfg=cfg, triplet_slice=triplet_slice, documents=documents, queries=queries, **kwargs)

if __name__ == '__main__':
    main()