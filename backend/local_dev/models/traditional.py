import numpy as np
import logging
import time

class SimpleDecisionTree:
    class Node:
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.value = None
            
    def __init__(self, max_depth=3, min_samples_split=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = self.Node()
        self.n_features_to_consider = 20  # Max number of features to consider at each split
        
    def _best_split(self, X, y):
        start_time = time.time()
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # Randomly select features to consider
        n_features_to_use = min(self.n_features_to_consider, n_features)
        feature_indices = np.random.choice(n_features, n_features_to_use, replace=False)
        
        # Calculate current impurity once
        if isinstance(y[0], (int, np.integer)):  # Classification
            classes, counts = np.unique(y, return_counts=True)
            current_impurity = 1 - np.sum((counts / n_samples) ** 2)
        else:  # Regression
            current_impurity = np.var(y)
        
        for feature in feature_indices:
            # Sample threshold values
            feature_values = np.unique(X[:, feature])
            if len(feature_values) > 10:
                feature_values = np.percentile(feature_values, np.linspace(10, 90, 10))
            
            for threshold in feature_values:
                left_mask = X[:, feature] <= threshold
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left
                
                if n_left < self.min_samples_split or n_right < self.min_samples_split:
                    continue
                
                y_left = y[left_mask]
                y_right = y[~left_mask]
                
                if isinstance(y[0], (int, np.integer)):
                    # Optimized classification impurity calculation
                    left_classes, left_counts = np.unique(y_left, return_counts=True)
                    right_classes, right_counts = np.unique(y_right, return_counts=True)
                    left_impurity = 1 - np.sum((left_counts / n_left) ** 2)
                    right_impurity = 1 - np.sum((right_counts / n_right) ** 2)
                else:
                    # Optimized regression impurity calculation
                    left_impurity = np.var(y_left) if len(y_left) > 1 else 0
                    right_impurity = np.var(y_right) if len(y_right) > 1 else 0
                
                gain = current_impurity - (n_left/n_samples * left_impurity + n_right/n_samples * right_impurity)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if time.time() - start_time > 0.1:  # Log if split takes more than 0.1 seconds
            logging.debug(f"Split took {time.time() - start_time:.2f} seconds")
            
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, node, depth):
        n_samples = len(y)
        
        # Early stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            node.value = np.mean(y) if isinstance(y[0], (float, np.float64)) else np.bincount(y).argmax()
            return
        
        feature, threshold = self._best_split(X, y)
        if feature is None:  # No valid split found
            node.value = np.mean(y) if isinstance(y[0], (float, np.float64)) else np.bincount(y).argmax()
            return
        
        node.feature = feature
        node.threshold = threshold
        
        # Split the data
        left_mask = X[:, feature] <= threshold
        
        # Create child nodes
        node.left = self.Node()
        node.right = self.Node()
        
        # Recursively build the tree
        self._build_tree(X[left_mask], y[left_mask], node.left, depth + 1)
        self._build_tree(X[~left_mask], y[~left_mask], node.right, depth + 1)
    
    def fit(self, X, y):
        logging.info(f"Starting decision tree training with data shape: {X.shape}")
        start_time = time.time()
        self._build_tree(X, y, self.root, 0)
        training_time = time.time() - start_time
        logging.info(f"Decision tree training completed in {training_time:.2f} seconds")
        return self
    
    def _predict_single(self, x, node):
        if node.value is not None:  # Leaf node
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)
    
    def predict(self, X):
        start_time = time.time()
        predictions = np.array([self._predict_single(x, self.root) for x in X])
        prediction_time = time.time() - start_time
        logging.info(f"Predictions completed in {prediction_time:.2f} seconds for {len(X)} samples")
        return predictions

class SimpleKNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Get k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            if isinstance(self.y_train[0], (int, np.integer)):  # Classification
                # Majority vote
                prediction = np.bincount(k_nearest_labels).argmax()
            else:  # Regression
                # Mean of nearest neighbors
                prediction = np.mean(k_nearest_labels)
                
            predictions.append(prediction)
        
        return np.array(predictions)
    
class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # For classification, use mode; for regression, use mean
        if isinstance(predictions[0][0], (int, np.integer)):
            return np.array([np.bincount(pred).argmax() for pred in predictions.T])
        return np.mean(predictions, axis=0)
    
# Add these clustering classes to your traditional.py file

class SimpleKMeans:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.centroids = None
        self.labels_ = None
        
    def fit(self, X):
        n_samples = X.shape[0]
        # Randomly initialize centroids
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_samples)[:self.n_clusters]
        self.centroids = X[indices]
        
        for _ in range(100):  # Max iterations
            # Assign points to nearest centroid
            distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids) ** 2).sum(axis=2))
            new_labels = np.argmin(distances, axis=1)
            
            # Update centroids
            old_centroids = self.centroids.copy()
            for k in range(self.n_clusters):
                if np.sum(new_labels == k) > 0:
                    self.centroids[k] = X[new_labels == k].mean(axis=0)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                break
                
        self.labels_ = new_labels
        return self
    
    def predict(self, X):
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

class SimpleDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        
    def fit(self, X):
        self.labels_ = self.fit_predict(X)
        return self
    
    def predict(self, X):
        # For DBSCAN, predict just returns the labels from fit
        return self.labels_
    
    def fit_predict(self, X):
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)  # -1 represents noise points
        cluster_id = 0
        
        # Calculate distance matrix
        distances = np.sqrt(((X[:, np.newaxis, :] - X) ** 2).sum(axis=2))
        
        for i in range(n_samples):
            if labels[i] != -1:
                continue
                
            neighbors = np.where(distances[i] <= self.eps)[0]
            
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Mark as noise
                continue
                
            # Start a new cluster
            cluster_id += 1
            labels[i] = cluster_id
            
            # Expand cluster
            neighbors = list(neighbors)
            while neighbors:
                current_point = neighbors.pop()
                if labels[current_point] == -1:
                    labels[current_point] = cluster_id
                    
                    new_neighbors = np.where(distances[current_point] <= self.eps)[0]
                    if len(new_neighbors) >= self.min_samples:
                        neighbors.extend([x for x in new_neighbors if labels[x] == -1])
        
        return labels

class SimpleAgglomerativeClustering:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.labels_ = None
        
    def fit_predict(self, X):
        n_samples = X.shape[0]
        # Initialize each point as its own cluster
        labels = np.arange(n_samples)
        distances = np.sqrt(((X[:, np.newaxis, :] - X) ** 2).sum(axis=2))
        
        current_n_clusters = n_samples
        while current_n_clusters > self.n_clusters:
            # Find closest pair of clusters
            mask = np.where(~np.eye(len(distances), dtype=bool))
            i, j = np.unravel_index(np.argmin(distances[mask]), (len(distances), len(distances)))
            if i > j:
                i, j = j, i
                
            # Merge clusters
            labels[labels == labels[j]] = labels[i]
            labels[labels > labels[j]] -= 1
            
            # Update distances (using single linkage)
            new_distances = np.minimum(distances[i], distances[j])
            distances = np.delete(np.delete(distances, j, axis=0), j, axis=1)
            distances[i] = new_distances
            distances[:, i] = new_distances
            
            current_n_clusters -= 1
            
        # Relabel clusters from 0 to n_clusters-1
        unique_labels = np.unique(labels)
        self.labels_ = np.zeros_like(labels)
        for i, label in enumerate(unique_labels):
            self.labels_[labels == label] = i
            
        return self.labels_
    
    def fit(self, X):
        self.fit_predict(X)
        return self