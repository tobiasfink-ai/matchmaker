import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import random

# Create Triplets for dense retrieval training.


def document_iterator(document_file, n_docs, batch_size:int=None):
    """
    Iterate through documents, returning only their text.
    Yields single documents if batch_size is zero or none.
    Yields lists of documents if batch_size is set.
    """
    with open(document_file, "r") as fp:
        if not batch_size or batch_size <= 0: # single document version
            for doc_line in tqdm(fp, total=n_docs):
                data = doc_line.strip().split(sep="\t")
                if len(data) != 4:
                    #print(len(data), data)
                    continue
                doc_id, doc_url, doc_title, doc_text = data
                yield doc_text
        else: # batched version
            docs_batch = []
            for doc_line in tqdm(fp, total=n_docs):
                data = doc_line.strip().split(sep="\t")
                if len(data) != 4:
                    #print(len(data), data)
                    continue
                doc_id, doc_url, doc_title, doc_text = data
                docs_batch.append(doc_text)
                if len(docs_batch) >= batch_size:
                    yield docs_batch
                    docs_batch = []
            if len(docs_batch) > 0:
                yield docs_batch


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
        for doc_line in tqdm(fp, total=n_docs):
            data = doc_line.strip().split(sep="\t")
            if len(data) != 4:
                #print(len(data), data)
                continue
            doc_id, doc_url, doc_title, doc_text = data
            documents[doc_id] = doc_text
    return documents


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


@hydra.main(version_base=None, config_path=".", config_name=None)
def main(cfg):
    n_docs = cfg.n_docs
    document_file = cfg.document_file
    queries_file = cfg.queries_file
    qrels_file = cfg.qrels_file
    triplet_folder = cfg.triplet_folder
    top_k = cfg.index_config.top_k

    random_seed = cfg.random_seed
    truncate = cfg.truncate
    analysis = cfg.analysis

    triplet_txt_file = os.path.join(triplet_folder, f"triplets_random-baseline.train.txt")

    doc_ids, doc_ids_inv, doc_ids_int64 = get_document_ids(document_file)
    documents = get_documents(document_file, n_docs=n_docs)
    queries = get_queries(queries_file)
    relevance_judgements = read_qrels(qrels_file)

    print("Creating Triplets")
    triplets = []
    for q_id, pos_doc_id, relevant_doc_ids in tqdm(relevance_judgement_iter(relevance_judgements, doc_ids_inv)):
        # return up to top_k negatives, filtering out relevant doc ids for the given relevance judgements
        neg_doc_ids = random.sample(doc_ids, k=top_k)
        for neg_doc_id in neg_doc_ids:
            if neg_doc_id in relevant_doc_ids:
                continue
            triplets.append((q_id, pos_doc_id, neg_doc_id))
    

    random.seed(random_seed)

    # shuffle training data
    print("Shuffling Triplets")
    random.shuffle(triplets)

    print("Writing Triplets")
    with open(os.path.join(triplet_folder, triplet_txt_file), 'w') as fp_out:
        for q_id, pos_doc_id, neg_doc_id in tqdm(triplets):
            pos_doc = documents[pos_doc_id]
            neg_doc = documents[neg_doc_id]
            if truncate:
                pos_doc = " ".join(pos_doc.split()[:512])
                neg_doc = " ".join(neg_doc.split()[:512])
            if analysis:
                fp_out.write(f"{queries[q_id]}\n===\n{pos_doc}\n===\n{neg_doc}\n\n\n")
            else:
                fp_out.write(f"{queries[q_id]}\t{pos_doc}\t{neg_doc}\n")
    


if __name__ == '__main__':
    main()