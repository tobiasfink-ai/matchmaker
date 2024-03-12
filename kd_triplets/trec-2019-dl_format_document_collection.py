import os
import json
import argparse
from tqdm import tqdm
import time
import random
random.seed(42)


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
    formatted_document_file = args.formatted_document_file

    with open(document_file, "r") as fp, open(formatted_document_file, "w") as out_fp:
        for doc_line in tqdm(fp, total=n_docs):
            data = doc_line.strip().split(sep="\t")
            if len(data) != 4:
                #print(len(data), data)
                continue
            doc_id, doc_url, doc_title, doc_text = data
            out_fp.write(f"{doc_id}\t{doc_text}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='From the document file create a file with document format: doc-id<tab>doc-text.'
    )

    parser.add_argument('--document_file', help='TSV file with document data')
    parser.add_argument('--formatted_document_file', help='TSV file with formatted document data')
    parser.add_argument('--n_docs', default=3213835, type=int, help='Number of documents')

    args = parser.parse_args()
    main(args)