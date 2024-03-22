from tqdm import tqdm
import ir_datasets


class Dataloader:

    def document_iterator(self):
        pass


    def get_documents(self):
        pass


    def get_document_ids(self):
        pass


    def read_qrels(self):
        pass


    def get_queries(self):
        pass




class TrecDLLoader(Dataloader):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.document_file = cfg.document_file
        self.n_docs = cfg.n_docs
        self.batch_size = cfg.batch_size
        self.qrels_file = cfg.qrels_file
        self.queries_file = cfg.queries_file
    

    def document_iterator(self):
        """
        Iterate through documents, returning only their text.
        Yields single documents if batch_size is zero or none.
        Yields lists of documents if batch_size is set.
        """
        with open(self.document_file, "r") as fp:
            if not self.batch_size or self.batch_size <= 0: # single document version
                for doc_line in tqdm(fp, total=self.n_docs):
                    data = doc_line.strip().split(sep="\t")
                    if len(data) != 4:
                        #print(len(data), data)
                        continue
                    doc_id, doc_url, doc_title, doc_text = data
                    yield doc_text
            else: # batched version
                docs_batch = []
                for doc_line in tqdm(fp, total=self.n_docs):
                    data = doc_line.strip().split(sep="\t")
                    if len(data) != 4:
                        #print(len(data), data)
                        continue
                    doc_id, doc_url, doc_title, doc_text = data
                    docs_batch.append(doc_text)
                    if len(docs_batch) >= self.batch_size:
                        yield docs_batch
                        docs_batch = []
                if len(docs_batch) > 0:
                    yield docs_batch


    def get_documents(self):
        """
        Returns the content of all documents as a list of strings
        """
        documents = {}
        with open(self.document_file, "r") as fp:
            for doc_line in tqdm(fp, total=self.n_docs, desc="Load Documents"):
                data = doc_line.strip().split(sep="\t")
                if len(data) != 4:
                    #print(len(data), data)
                    continue
                doc_id, doc_url, doc_title, doc_text = data
                documents[doc_id] = doc_text
        return documents


    def get_document_ids(self):
        """
        Returns:
          - doc_ids: a list of document ids as they are in the collection
          - doc_ids_inv: a mapping from internal doc id to collection doc id
          - doc_ids_int64: a list of internal document ids in int64 format
        """
        doc_ids = []
        doc_ids_inv = {}
        doc_ids_int64 = []
        with open(self.document_file, "r") as fp:
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


    def read_qrels(self):
        """
        Returns a dictionary of relevance judgements in the form of query_id: list of relevant document_ids
        """
        relevance_judgements = {}
        with open(self.qrels_file, "r") as fp:
            for qrel_line in fp:
                q_id, _, doc_id, relevance = qrel_line.strip().split()
                relevance = int(relevance)
                if relevance == 1:
                    if q_id not in relevance_judgements:
                        relevance_judgements[q_id] = set()
                    relevance_judgements[q_id].add(doc_id)
        return relevance_judgements


    def get_queries(self):
        """
        Returns a dictionary of query_id: query_text
        """
        queries = {}
        with open(self.queries_file, "r") as fp:
            for queries_line in fp:
                q_id, q_text = queries_line.strip().split(sep="\t")
                queries[q_id] = q_text
        return queries



class TrecRobust04Loader(Dataloader):
    """
    Requires a manual download of TREC disk4 and 5 containing the robust data
    Store this data in $IR_DATASETS_HOME/disc45/corpus
    By default IR_DATASETS_HOME is set to ~/.ir_datasets
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.dataset_name = "disks45/nocr/trec-robust-2004"
        self.batch_size = cfg.batch_size
        self.folds = cfg.folds # qrels and queries from only the selected folds


    def document_iterator(self):
        """
        Iterate through documents, returning only their text.
        Yields single documents if batch_size is zero or none.
        Yields lists of documents if batch_size is set.
        """
        dataset = ir_datasets.load(self.dataset_name)
        if not self.batch_size or self.batch_size <= 0: # single document version
            for doc in tqdm(dataset.docs_iter()):
                doc_text = f"{doc.title} {doc.body}"
                yield doc_text
        else: # batched version
            docs_batch = []
            for doc in tqdm(dataset.docs_iter()):
                doc_text = f"{doc.title} {doc.body}"
                docs_batch.append(doc_text)
                if len(docs_batch) >= self.batch_size:
                    yield docs_batch
                    docs_batch = []
            if len(docs_batch) > 0:
                yield docs_batch


    def get_documents(self):
        """
        Returns the content of all documents as a list of strings
        """
        documents = {}
        dataset = ir_datasets.load(self.dataset_name)
        for doc in tqdm(dataset.docs_iter()):
            doc_text = f"{doc.title} {doc.body}"
            documents[doc.doc_id] = doc_text
        return documents


    def get_document_ids(self):
        """
        Returns:
          - doc_ids: a list of document ids as they are in the collection
          - doc_ids_inv: a mapping from internal doc id to collection doc id
          - doc_ids_int64: a list of internal document ids in int64 format
        """
        doc_ids = []
        doc_ids_inv = {}
        doc_ids_int64 = []
        dataset = ir_datasets.load(self.dataset_name)
        for doc in tqdm(dataset.docs_iter()):
            doc_id = doc.doc_id
            doc_ids_inv[doc_id] = len(doc_ids)
            doc_ids_int64.append(len(doc_ids))
            doc_ids.append(doc_id)
        return doc_ids, doc_ids_inv, doc_ids_int64


    def read_qrels(self):
        """
        Returns a dictionary of relevance judgements in the form of query_id: list of relevant document_ids
        """
        relevance_judgements = {}
        fold_qrels = set()
        # collect all qrels of the selected folds
        for fold in self.folds:
            dataset = ir_datasets.load(f"{self.dataset_name}/fold{fold}")
            for qrel in dataset.qrels_iter():
                fold_qrels.add(qrel)
        # create relevance_judgements with all relevance >=1 documents 
        for qrel in fold_qrels:
            # query_id, doc_id, relevance
            relevance = int(qrel.relevance)
            if relevance >= 1:
                if qrel.query_id not in relevance_judgements:
                    relevance_judgements[qrel.query_id] = set()
                relevance_judgements[qrel.query_id].add(qrel.doc_id)
        return relevance_judgements


    def get_queries(self):
        """
        Returns a dictionary of query_id: query_text
        """
        queries = {}
        fold_queries = set()
        # collect all queries of the selected folds
        for fold in self.folds:
            dataset = ir_datasets.load(f"{self.dataset_name}/fold{fold}")
            for query in dataset.queries_iter():
                fold_queries.add(query)
        for query in fold_queries:
            # query_id, title, description, narrative
            q_text = f"{query.title} {query.description}"
            queries[query.query_id] = q_text
        return queries
