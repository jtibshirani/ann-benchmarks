from __future__ import absolute_import
import array, sys
import numpy
import sklearn.neighbors
import sklearn.preprocessing

from ann_benchmarks.distance import metrics as pd
from ann_benchmarks.algorithms.base import BaseANN

from py4j.java_collections import ListConverter
from py4j.java_gateway import JavaGateway

class LuceneCluster(BaseANN):
    INDEX_BATCH_SIZE = 1000

    def __init__(self, metric):
        if metric not in ('angular', 'euclidean'):
            raise NotImplementedError(
                "LuceneCluster doesn't support metric %s" % metric)
        self._metric = metric
        self.gateway = JavaGateway()

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        self.gateway.entry_point.prepareIndex()

        start = 0
        while start < X.shape[0]:
            end = min(start + self.INDEX_BATCH_SIZE, X.shape[0])
            batch = self.prepare_vectors(X[start:end])
            self.gateway.entry_point.indexBatch(start, batch)
            start = end
            print("Finished indexing {} vectors".format(end))

        self.gateway.entry_point.mergeAndCommit()
        self.gateway.entry_point.openReader()

    def batch_query(self, X, n):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        query_vectors = self.prepare_vectors(X)
        self.res = self.gateway.entry_point.search(query_vectors, n, self.n_probes)

    def get_batch_results(self):
        return self.res

    def set_query_arguments(self, n_probes):
        self.n_probes = n_probes

    def __str__(self):
        return 'LuceneCluster(n_probes={})'.format(self.n_probes)

    def prepare_vectors(self, vectors):
        header = array.array('i', list(vectors.shape))
        body = array.array('f', vectors.flatten().tolist());
        if sys.byteorder != 'big':
            header.byteswap()
            body.byteswap()
        return bytearray(header.tostring() + body.tostring())
