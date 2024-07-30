import numpy as np
import operator

class KNN(object):

    def __init__(self):
        pass
        
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)
        
    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):   
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum((X[i, :] - self.X_train[j, :]) ** 2))
        return dists
        
    def predict_labels(self, dists, k=1):
 
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            k_nearest_index = np.argsort(dists[i, :])[:k]
            closest_y = self.y_train[k_nearest_index]
            labels_counts = {}
            for label in closest_y:
                if label in labels_counts.keys():
                    labels_counts[label] += 1
                else:
                    labels_counts[label] = 0
            sorted_labels_counts = sorted(
                labels_counts.items(), key=operator.itemgetter(1), reverse=True)
            y_pred[i] = sorted_labels_counts[0][0]
        return y_pred