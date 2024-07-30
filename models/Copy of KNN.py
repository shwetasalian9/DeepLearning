import numpy as np
import scipy
from collections import Counter

def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


class KNN():
    def __init__(self, k):
        self.k = k
    
    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        
        self.X_train = X
        self.y_train = y
    
    def find_dist(self, X_test):
        dist_ = [euclidean_distance(X_test, x_train) for x_train in self.X_train]
        return dist_
    
    def predict(self, X_test):
        distances = self.find_dist(X_test)
        k_idx = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        return Counter(k_neighbor_labels).most_common(1)
        