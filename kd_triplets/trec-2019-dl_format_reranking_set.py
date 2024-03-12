import os
import json
import argparse
from tqdm import tqdm
import time


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


def main(args):
    n_docs = args.n_docs
    document_file = args.document_file
    queries_file = args.queries_file
    top_n_id_file = args.top_n_id_file
    top_n_text_file = args.top_n_text_file
    truncate = args.truncate

    documents = get_documents(document_file, n_docs=n_docs)
    queries = get_queries(queries_file)

    print("Creating Reranking Data")
    with open(top_n_id_file, 'r') as fp, open(top_n_text_file, 'w') as fp_out:
        for line in fp:
            q_id, _, doc_id, rank, s, _ = line.strip().split()
            query = queries[q_id]
            if doc_id not in documents:
                continue
            doc = documents[doc_id]
            if truncate:
                doc = " ".join(doc.split()[:512])
            fp_out.write(f"{q_id}\t{doc_id}\t{query}\t{doc}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='From the top n file create re-ranking validation data for dense retrieval training in the format: query-id<tab>doc-id<tab>query-text<tab>doc-text.'
    )

    parser.add_argument('--document_file', help='TSV file with document data')
    parser.add_argument('--queries_file', help='File with queries data')
    parser.add_argument('--top_n_id_file', help='File with query ids and top N document ids')
    parser.add_argument('--top_n_text_file', help='Out File with query texts and top N document texts')
    parser.add_argument('-t', '--truncate',
                    action='store_true')
    parser.add_argument('--n_docs', default=3213835, type=int, help='Number of documents')

    args = parser.parse_args()
    main(args)