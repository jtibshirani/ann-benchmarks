from __future__ import absolute_import
import numpy
import sklearn.neighbors
import sklearn.preprocessing

from ann_benchmarks.distance import metrics as pd
from ann_benchmarks.algorithms.base import BaseANN

from py4j.java_collections import ListConverter
from py4j.java_gateway import JavaGateway

class LuceneGraph(BaseANN):
    INDEX_BATCH_SIZE = 1000

    def __init__(self, metric):
        if metric not in ('angular', 'euclidean'):
            raise NotImplementedError(
                "LuceneGraph doesn't support metric %s" % metric)
        self._metric = metric
        self.gateway = JavaGateway()

    def fit(self, X):
        function = 'EUCLIDEAN' if self._metric is 'euclidean' else 'COSINE'
        self.gateway.entry_point.prepareIndex(function)

        start = 0
        while start < X.shape[0]:
            end = min(start + self.INDEX_BATCH_SIZE, X.shape[0])
            batch = self.prepare_vectors(X[start:end])
            self.gateway.entry_point.indexBatch(start, batch)
            start = end

        self.gateway.entry_point.forceMerge()
        self.gateway.entry_point.openReader()

    def query(self, v, n):
        query_vector = self.prepare_vector(v)
        return self.gateway.entry_point.search(query_vector, n, self.ef)

    def set_query_arguments(self, ef):
        self.ef = ef

    def __str__(self):
        return 'LuceneGraph(M=6, ef_const=50, ef={})'.format(self.ef)

    def prepare_vectors(self, vectors):
        converted_vectors = [self.prepare_vector(v) for v in vectors]
        return ListConverter().convert(converted_vectors, self.gateway._gateway_client)

    def prepare_vector(self, vector):
        return ListConverter().convert(vector.tolist(), self.gateway._gateway_client)
