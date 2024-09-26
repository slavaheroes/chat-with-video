from time import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import metrics
import minsearch
from loguru import logger
from elasticsearch import Elasticsearch
from generate_datasets import VIDEOS
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
es_client = Elasticsearch('http://localhost:9200')


class CustomVectorSearchEngine:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]


def tf_idf(df: pd.DataFrame) -> Tuple[List[List[bool]], float]:
    documents = []
    for chunk_id in df['chunk_id'].unique():
        doc = df[df['chunk_id'] == chunk_id].iloc[0]['chunk']
        documents.append({'id': chunk_id, 'text': doc})

    index = minsearch.Index(text_fields=['text'], keyword_fields=['id'])
    index.fit(documents)
    start = time()

    relevance_total = []
    for i, row in df.iterrows():
        doc_id = row['chunk_id']
        query = df.iloc[i]['question']
        results = index.search(query=query, num_results=10)
        relevance = relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    latency = time() - start

    return relevance_total, latency


def custom_vector_search(df: pd.DataFrame) -> Tuple[List[List[bool]], float]:
    documents = []
    for chunk_id in df['chunk_id'].unique():
        doc = df[df['chunk_id'] == chunk_id].iloc[0]['chunk']
        documents.append({'id': chunk_id, 'text': doc})

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode([doc['text'] for doc in documents])
    logger.info(f'Custom vector search: {len(embeddings)} embeddings with dimension {len(embeddings[0])}')

    search_engine = CustomVectorSearchEngine(documents, embeddings)
    start = time()
    relevance_total = []
    for i, row in df.iterrows():
        doc_id = row['chunk_id']
        query = df.iloc[i]['question']
        v_query = embedder.encode(query)
        results = search_engine.search(v_query, num_results=10)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    latency = time() - start
    return relevance_total, latency


def default_es(df: pd.DataFrame) -> Tuple[List[List[bool]], float]:
    documents = []
    for chunk_id in df['chunk_id'].unique():
        doc = df[df['chunk_id'] == chunk_id].iloc[0]['chunk']
        documents.append({'id': chunk_id, 'text': doc})

    index_name = 'default_es_eval'
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {"text": {"type": "text"}, "id": {"type": "keyword"}}},
    }

    es_client.indices.create(index=index_name, body=index_settings)

    for doc in documents:
        es_client.index(index=index_name, document=doc)

    start = time()
    relevance_total = []
    for i, row in df.iterrows():
        search_query = {
            "size": 10,
            "query": {
                "bool": {
                    "must": {"multi_match": {"query": row['question'], "fields": ["text"], "type": "best_fields"}},
                }
            },
        }

        response = es_client.search(index=index_name, body=search_query)
        results = response['hits']['hits']
        relevance = [d['_source']['id'] == row['chunk_id'] for d in results]
        relevance_total.append(relevance)

    latency = time() - start
    es_client.indices.delete(index=index_name)
    return relevance_total, latency


def hybrid_es(df: pd.DataFrame) -> Tuple[List[List[bool]], float]:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    documents = []
    for chunk_id in df['chunk_id'].unique():
        doc = df[df['chunk_id'] == chunk_id].iloc[0]['chunk']
        documents.append({'id': chunk_id, 'text': doc, 'vector': embedder.encode(doc)})

    index_name = 'hybrid_es_eval'
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "text": {"type": "text"},
                "id": {"type": "keyword"},
                "vector": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
            }
        },
    }

    es_client.indices.create(index=index_name, body=index_settings)

    for doc in documents:
        es_client.index(index=index_name, document=doc)

    start = time()
    relevance_total = []
    for i, row in df.iterrows():

        query = row['question']
        q_vector = embedder.encode(query)

        knn_query = {
            "field": 'vector',
            "query_vector": q_vector,
            "k": 10,
            "num_candidates": 10000,
            "boost": 0.5,
        }

        keyword_query = {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text"],
                        "type": "best_fields",
                        "boost": 0.5,
                    }
                }
            }
        }

        search_query = {"knn": knn_query, "query": keyword_query, "size": 10, "_source": ["text", "id"]}

        results = es_client.search(index=index_name, body=search_query)['hits']['hits']
        relevance = [d['_source']['id'] == row['chunk_id'] for d in results]
        relevance_total.append(relevance)

    latency = time() - start
    es_client.indices.delete(index=index_name)
    return relevance_total, latency


if __name__ == '__main__':
    evaluate_results = []

    for method in ['hybrid_es', 'default_es', 'tf_idf', 'custom_vdb']:

        for key in VIDEOS.keys():
            df_path = f'data/{key}_gt.csv'
            df = pd.read_csv(df_path)

            logger.info(f'Evaluating {key} with method {method}')
            if method == 'default_es':
                logger.info('Evaluating default ES')
                relevance_total, latency = default_es(df)
            elif method == 'tf_idf':
                logger.info('Evaluating tfidf')
                relevance_total, latency = tf_idf(df)
            elif method == 'custom_vdb':
                logger.info('Evaluating custom VDB')
                relevance_total, latency = custom_vector_search(df)
            elif method == 'hybrid_es':
                logger.info('Evaluating hybrid ES')
                relevance_total, latency = hybrid_es(df)

            mins, secs = divmod(latency, 60)
            logger.info(f'Latency: {mins:.0f}m {secs:.0f}s')

            hit_rate = metrics.hit_rate(relevance_total)
            mrr = metrics.mrr(relevance_total)

            evaluate_results.append({'method': method, 'video': key, 'hit_rate': hit_rate, 'mrr': mrr, 'latency': latency})

            logger.info(f'Hit rate: {hit_rate}')
            logger.info(f'MRR: {mrr}')

    df_results = pd.DataFrame(evaluate_results)
    df_results.to_csv('results/retrieval_evaluate_results.csv', index=False)
    print(df_results.to_string(index=False))
