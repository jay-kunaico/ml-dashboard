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
        clusters = [{i} for i in range(n_samples)]
        distances = np.sqrt(((X[:, np.newaxis, :] - X) ** 2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)

        while len(clusters) > self.n_clusters:
            # Find the closest pair of clusters
            min_dist = np.inf
            merge_pair = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Single linkage: minimum distance between points in clusters
                    dist = np.min([distances[p1, p2] for p1 in clusters[i] for p2 in clusters[j]])
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (i, j)
            i, j = merge_pair
            clusters[i].update(clusters[j])
            del clusters[j]

        labels = np.empty(n_samples, dtype=int)
        for idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = idx
        self.labels_ = labels
        return labels

    def fit(self, X):
        self.fit_predict(X)
        return self

    def predict(self, X):
        return self.labels_

class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.classes_ = None
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
        
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize weights and bias
        if n_classes == 2:
            self.weights = np.zeros(n_features)
            self.bias = 0
        else:
            self.weights = np.zeros((n_classes, n_features))
            self.bias = np.zeros(n_classes)
        
        # Convert y to one-hot encoding for multi-class
        if n_classes > 2:
            y_one_hot = np.zeros((n_samples, n_classes))
            for i, cls in enumerate(self.classes_):
                y_one_hot[y == cls, i] = 1
            y = y_one_hot
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Forward pass
            if n_classes == 2:
                z = np.dot(X, self.weights) + self.bias
                predictions = self._sigmoid(z)
                # Binary cross-entropy loss
                loss = -np.mean(y * np.log(predictions + 1e-15) + 
                              (1 - y) * np.log(1 - predictions + 1e-15))
            else:
                z = np.dot(X, self.weights.T) + self.bias
                predictions = self._softmax(z)
                # Categorical cross-entropy loss
                loss = -np.mean(np.sum(y * np.log(predictions + 1e-15), axis=1))
            
            # Backward pass
            if n_classes == 2:
                dw = np.dot(X.T, (predictions - y)) / n_samples
                db = np.mean(predictions - y)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            else:
                dw = np.dot((predictions - y).T, X) / n_samples
                db = np.mean(predictions - y, axis=0)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Check convergence
            if loss < self.tol:
                break
                
        return self
    
    def predict_proba(self, X):
        if len(self.classes_) == 2:
            z = np.dot(X, self.weights) + self.bias
            proba = self._sigmoid(z)
            return np.column_stack((1 - proba, proba))
        else:
            z = np.dot(X, self.weights.T) + self.bias
            return self._softmax(z)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            return (proba[:, 1] >= 0.5).astype(int)
        return self.classes_[np.argmax(proba, axis=1)]

class SimpleXGBoost:
    class TreeNode:
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.value = None
            self.weight = None
            
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                 reg_lambda=1.0, reg_alpha=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda  # L2 regularization
        self.reg_alpha = reg_alpha    # L1 regularization
        self.trees = []
        self.base_score = None
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
    def _compute_gradients(self, y_true, y_pred, task='binary'):
        if task == 'binary':
            p = self._sigmoid(y_pred)
            grad = p - y_true
            hess = p * (1 - p)
        else:  # regression
            grad = y_pred - y_true
            hess = np.ones_like(y_true)
        return grad, hess
        
    def _best_split(self, X, grad, hess, feature_indices):
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        n_samples = X.shape[0]
        total_grad = np.sum(grad)
        total_hess = np.sum(hess)
        
        for feature in feature_indices:
            # Get unique values for this feature
            thresholds = np.unique(X[:, feature])
            if len(thresholds) > 10:
                thresholds = np.percentile(thresholds, np.linspace(10, 90, 10))
                
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_child_weight or np.sum(right_mask) < self.min_child_weight:
                    continue
                    
                left_grad = np.sum(grad[left_mask])
                left_hess = np.sum(hess[left_mask])
                right_grad = np.sum(grad[right_mask])
                right_hess = np.sum(hess[right_mask])
                
                # Calculate gain
                left_score = (left_grad ** 2) / (left_hess + self.reg_lambda)
                right_score = (right_grad ** 2) / (right_hess + self.reg_lambda)
                total_score = (total_grad ** 2) / (total_hess + self.reg_lambda)
                
                gain = left_score + right_score - total_score
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
        
    def _build_tree(self, X, grad, hess, depth=0):
        node = self.TreeNode()
        
        if depth >= self.max_depth or X.shape[0] < self.min_child_weight:
            node.value = -np.sum(grad) / (np.sum(hess) + self.reg_lambda)
            return node
            
        # Select features for this tree
        n_features = X.shape[1]
        n_features_to_use = max(1, int(n_features * self.colsample_bytree))
        feature_indices = np.random.choice(n_features, n_features_to_use, replace=False)
        
        feature, threshold = self._best_split(X, grad, hess, feature_indices)
        
        if feature is None:
            node.value = -np.sum(grad) / (np.sum(hess) + self.reg_lambda)
            return node
            
        node.feature = feature
        node.threshold = threshold
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        node.left = self._build_tree(X[left_mask], grad[left_mask], hess[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], grad[right_mask], hess[right_mask], depth + 1)
        
        return node
        
    def _predict_tree(self, x, tree):
        if tree.value is not None:
            return tree.value
            
        if x[tree.feature] <= tree.threshold:
            return self._predict_tree(x, tree.left)
        return self._predict_tree(x, tree.right)
        
    def fit(self, X, y, task='binary'):
        n_samples = X.shape[0]
        self.base_score = np.mean(y) if task == 'regression' else 0
        y_pred = np.full(n_samples, self.base_score)
        
        for _ in range(self.n_estimators):
            # Compute gradients
            grad, hess = self._compute_gradients(y, y_pred, task)
            
            # Subsample data
            if self.subsample < 1.0:
                indices = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                X_sub = X[indices]
                grad_sub = grad[indices]
                hess_sub = hess[indices]
            else:
                X_sub = X
                grad_sub = grad
                hess_sub = hess
                
            # Build tree
            tree = self._build_tree(X_sub, grad_sub, hess_sub)
            self.trees.append(tree)
            
            # Update predictions
            for i in range(n_samples):
                y_pred[i] += self.learning_rate * self._predict_tree(X[i], tree)
                
        return self
        
    def predict(self, X, task='binary'):
        n_samples = X.shape[0]
        y_pred = np.full(n_samples, self.base_score)
        
        for tree in self.trees:
            for i in range(n_samples):
                y_pred[i] += self.learning_rate * self._predict_tree(X[i], tree)
                
        if task == 'binary':
            return (self._sigmoid(y_pred) >= 0.5).astype(int)
        return y_pred
        
    def predict_proba(self, X):
        n_samples = X.shape[0]
        y_pred = np.full(n_samples, self.base_score)
        
        for tree in self.trees:
            for i in range(n_samples):
                y_pred[i] += self.learning_rate * self._predict_tree(X[i], tree)
                
        proba = self._sigmoid(y_pred)
        return np.column_stack((1 - proba, proba))