
from typing import List

import faiss
import numpy
import pysparnn.cluster_index as ci


class BaseNNIndexer():
    '''
    Base class for our nearest neighbor indexing operations, atm we mainly abstrcat faiss, but it should allow us to swap in other libs fairly easy
    '''

    def __init__(self, config):
        super(BaseNNIndexer, self).__init__()

    def prepare(self, data_chunks:List[numpy.ndarray], subsample=-1):
        '''
        Train an index with (all) or only some vectors, if subsample is set to a value between 0 and 1
        '''
        pass

    def index(self, ids:List[numpy.ndarray], data_chunks:List[numpy.ndarray]):
        '''
        ids: need to be int64
        '''
        pass

    def search(self, query_vec:numpy.ndarray, top_k:int):
        '''
        query_vec: can be 2d (batch search) or 1d (single search) 
        '''
        pass


class PySparnnIndexer(BaseNNIndexer):

    def __init__(self, config):
        super(PySparnnIndexer, self).__init__(config)

        self.k_clusters = config["k_clusters"]

        self.pysparnn_index:ci.MultiClusterIndex = None

    def index(self, ids:List, embeddings):
        '''
        ids: need to be int64
        '''
        self.pysparnn_index = ci.MultiClusterIndex(embeddings, ids)

    def search(self, search_features_vec_batch, top_k:int):
        '''
        search_features_vec_batch: should be 2d (batch search)
        '''
        sims_with_idx = self.pysparnn_index.search(search_features_vec_batch, k=top_k, k_clusters=self.k_clusters, return_distance=True)
        for query_sims in sims_with_idx:
                for i in range(len(query_sims)):
                    distance, neg_doc_id = query_sims[i]
                    similarity = 1-float(distance)
                    query_sims[i] = (similarity, neg_doc_id)
        return sims_with_idx


class FaissBaseIndexer(BaseNNIndexer):
    '''
    Shared faiss code
    '''

    def __init__(self,config):
        super(FaissBaseIndexer, self).__init__(config)
        self.token_dim = config["token_dim"]
        self.use_gpu = config["faiss_use_gpu"]
        self.use_fp16 = config["token_dtype"] == "float16"
        self.faiss_index:faiss.Index = None # needs to be initialized by the actual faiss classes

    def index(self, ids:List[numpy.ndarray], embeddings):
        # single add needed for multi-gpu index (sharded), and hnsw so just do it for all (might be a memory problem at some point, but we can come back to that)
        i = numpy.array(ids, dtype=numpy.int64)
        #i = numpy.concatenate(ids).astype(numpy.int64)
        #c = numpy.concatenate(data_chunks).astype(numpy.float32)
        #console.log("[FaissIndexer]","Add",c.shape[0]," vectors")
        self.faiss_index.add_with_ids(embeddings,i)

    def search(self, query_vec:numpy.ndarray, top_k:int):
        # even a single search must be 1xn dims
        if len(query_vec.shape) == 1:
            query_vec = query_vec[numpy.newaxis,:]
            
        res_scores, indices = self.faiss_index.search(query_vec.astype(numpy.float32),top_k)
        
        sims_with_idx = []

        for similarities, ids in zip(res_scores,indices):
            query_sims = []
            sims_with_idx.append(query_sims)
            for s, i in zip(similarities, ids):
                query_sims.append((s,i))

        return sims_with_idx

    def save(self, path:str):
        if self.use_gpu:
            idx = faiss.index_gpu_to_cpu(self.faiss_index)
        else:
            idx = self.faiss_index
        faiss.write_index(idx, path)
    
    def load(self, path:str,config_overwrites=None):
        self.faiss_index = faiss.read_index(path)


class FaissIdIndexer(FaissBaseIndexer):
    '''
    Simple brute force nearest neighbor faiss index with id mappings, with potential gpu usage, support for fp16
    -> if faiss_use_gpu=True use all availbale GPUs in a sharded index 
    '''

    def __init__(self,config):
        super(FaissIdIndexer, self).__init__(config)

        if self.use_gpu:

            #console.log("[FaissIdIndexer]","Index on GPU")
            cpu_index = faiss.IndexIDMap(faiss.IndexFlatIP(config["token_dim"]))
                        
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = self.use_fp16

            self.faiss_index = faiss.index_cpu_to_all_gpus(cpu_index,co)

        else:
            #console.log("[FaissIdIndexer]","Index on CPU")
            if self.use_fp16:
                self.faiss_index = faiss.IndexIDMap(faiss.IndexScalarQuantizer(config["token_dim"],faiss.ScalarQuantizer.QT_fp16,faiss.METRIC_INNER_PRODUCT))
            else:
                self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(config["token_dim"]))


class FaissHNSWIndexer(FaissBaseIndexer):
    '''
    HNSW - graph based - index, only supports CPU - but gets very low query latency 
    '''

    def __init__(self,config):
        super(FaissHNSWIndexer, self).__init__(config)

        self.use_gpu = False # HNSW does not support GPUs

        if self.use_fp16:
            #console.log("[FaissHNSWIndexer]","Index with fp16")
            self.faiss_index = faiss.IndexHNSWSQ(config["token_dim"],faiss.ScalarQuantizer.QT_fp16,
                                                config["faiss_hnsw_graph_neighbors"],faiss.METRIC_INNER_PRODUCT)

        else:
            #console.log("[FaissHNSWIndexer]","Index with fp32")
            self.faiss_index = faiss.IndexHNSWFlat(config["token_dim"],config["faiss_hnsw_graph_neighbors"],faiss.METRIC_INNER_PRODUCT)
        
        self.faiss_index.verbose = True
        self.faiss_index.hnsw.efConstruction = config["faiss_hnsw_efConstruction"]
        self.faiss_index.hnsw.efSearch = config["faiss_hnsw_efSearch"]

        self.faiss_index = faiss.IndexIDMap(self.faiss_index)

    def prepare(self, data_chunks:List[numpy.ndarray], subsample=-1):
        if self.use_fp16:
            # training for the scalar quantizer, according to: https://github.com/facebookresearch/faiss/blob/master/benchs/bench_hnsw.py
            self.faiss_index.train(numpy.concatenate(data_chunks).astype(numpy.float32))