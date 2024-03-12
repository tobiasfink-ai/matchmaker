from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np


class TFIDFDelta:

    def __init__(self, kd_model_config) -> None:
        self.max_df = kd_model_config.tfidf_max_df
        self.min_df = kd_model_config.tfidf_min_df
        self.tfidf_vect = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df)
    

    def create_embeddings(self, document_iter):
        # expects a document_iter that returns single documents
        X = self.tfidf_vect.fit_transform(document_iter)
        return X


class BERTDelta:

    def __init__(self, kd_model_config) -> None:
        self.embedding_model = kd_model_config.embedding_model
        self.model = SentenceTransformer(self.embedding_model)
        self.model.max_seq_length = kd_model_config.max_seq_length
        self.embedding_batch_size = kd_model_config.embedding_batch_size
    

    def create_embeddings(self, document_iter):
        # expects a document_iter that returns batches
        data_chunks = []
        for doc_batch in document_iter:
            batch_embeddings = self.model.encode(doc_batch, batch_size=self.embedding_batch_size, convert_to_numpy=True)
            data_chunks.append(batch_embeddings)
        X = np.concatenate(data_chunks).astype(np.float32)
        return X