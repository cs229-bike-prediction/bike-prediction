import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, svm
from sklearn.cluster import KMeans, MiniBatchKMeans

class SeveralKMeans():
    def __init__(self, n_clusters, verbose=False):
        self.verbose = verbose
        self.n_clusters = n_clusters

        self.clusters = []
        for i in range(len(self.n_clusters)):
            self.clusters.append(KMeans(n_clusters=n_clusters[i]))

        self.estimators = []

    def fit(self, locs, cluster_ids):
        for i in range(len(self.n_clusters)):
            if self.verbose:
                print 'Fitting data for cluster_id=%d' % i
            training_locs = locs[cluster_ids==i]
            self.clusters[i].fit(training_locs)
            if self.verbose:
                print 'Finished training %s' % i

    def make_estimators(self, locs, X, y, cluster_ids):
        self.estimators = []
        for i in range(len(self.n_clusters)):
            # Need to make `self.n_clusters[i]` estimators for this cluster group
            es = []

            # First get the X and y we use to train this set of estimators, which
            # is the set of training samples which fall into cluster `i`
            training_X = X[cluster_ids==i]
            training_y = y[cluster_ids==i]
            training_labels = self.clusters[i].predict(locs[cluster_ids==i])
            plt.plot(np.histogram(training_labels, range(25))[0])

            for j in range(self.n_clusters[i]):
                if self.verbose:
                    print 'Building estimator in cluster_id=%d, cluster_label=%d' % (i, j)

                training_labels_j = training_labels==j
                if np.any(training_labels_j):
                    # Let's make an estimator for this cluster
                    clf = linear_model.LinearRegression()
                    clf.fit(training_X[training_labels_j,:], training_y[training_labels_j])
                    es.append(clf)
                else:
                    print 'No samples in training class j', j
                    es.append(None)

            self.estimators.append(es)

        plt.xlim((0, max(self.n_clusters)-1))
        plt.show()

    def predict(self, locs, X, cluster_ids):
        pr = np.zeros(cluster_ids.shape)

        # For each cluster_id, predict separately
        for i in range(len(self.n_clusters)):
            # Predict separately for each cluster
            tst_X = X[cluster_ids==i]
            tst_labels = self.clusters[i].predict(locs[cluster_ids==i])
            cpr = np.zeros(tst_labels.shape)

            for j in range(self.n_clusters[i]):
                if self.verbose:
                    print 'Predicting in cluster_id=%d, cluster_label=%d' % (i, j)
                cpr[tst_labels==j] = self.estimators[i][j].predict(tst_X[tst_labels==j])

            pr[cluster_ids==i] = cpr

        return pr
