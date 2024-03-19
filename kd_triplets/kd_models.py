from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import LongformerModel, AutoTokenizer
import numpy as np
import torch


class TFIDFDelta:

    def __init__(self, kd_model_config) -> None:
        self.max_df = kd_model_config.tfidf_max_df
        self.min_df = kd_model_config.tfidf_min_df
        self.tfidf_vect = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df)
    

    def create_embeddings(self, document_iter):
        # expects a document_iter that returns single documents
        X = self.tfidf_vect.fit_transform(document_iter)
        return X


class SentenceTransformerDelta:

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


class LongformerDelta:

    def __init__(self, kd_model_config) -> None:
        self.embedding_model = kd_model_config.embedding_model
        # Load model directly
        self.model = LongformerModel.from_pretrained(self.embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.max_seq_length = kd_model_config.max_seq_length
        self.global_attention_mask_tokens = kd_model_config.global_attention_mask_tokens

        self.model.cuda()
    

    def create_embeddings(self, document_iter):
        cuda_device = 0 # always take the first
        # expects a document_iter that returns batches
        data_chunks = []
        for doc_batch in document_iter:
            input_ids = self.tokenizer(text=doc_batch, 
                                       return_tensors="pt", 
                                       padding='longest', 
                                       max_length=self.max_seq_length, 
                                       truncation=True)
            input_ids = input_ids['input_ids'].cuda(cuda_device)
            attention_mask = torch.ones(
                input_ids.shape, dtype=torch.long, device=input_ids.device
            )  # initialize to local attention
            global_attention_mask = torch.zeros(
                input_ids.shape, dtype=torch.long, device=input_ids.device
            ) # initialize to global attention to be deactivated for all tokens
            global_attention_mask[
                :,
                :self.global_attention_mask_tokens,
            ] = 1  # Set global attention on the beginning of paragraphs
            outputs = self.model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
            pooled_output = outputs.pooler_output.cpu()
            data_chunks.append(pooled_output.detach().numpy())
        X = np.concatenate(data_chunks).astype(np.float32)
        return X