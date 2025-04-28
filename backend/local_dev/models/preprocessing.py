import numpy as np
import pandas as pd

class SimpleStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit_transform(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0] = 1  # Prevent division by zero
        return (data - self.mean) / self.std
    
    def transform(self, data):
        return (data - self.mean) / self.std


class SimpleOneHotEncoder:
    def __init__(self):
        self.categories = {}
        
    def fit_transform(self, data):
        # Convert DataFrame to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        unique_values = {}
        for col in range(data.shape[1]):
            unique_values[col] = list(set(data[:, col]))
            self.categories[col] = {val: i for i, val in enumerate(unique_values[col])}
        
        # Create one-hot matrix
        total_categories = sum(len(cats) for cats in unique_values.values())
        result = np.zeros((data.shape[0], total_categories))
        
        current_pos = 0
        for col in range(data.shape[1]):
            for val in data[:, col]:
                if val in self.categories[col]:
                    result[:, current_pos + self.categories[col][val]] = 1
            current_pos += len(unique_values[col])
            
        return result
    
    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        total_categories = sum(len(cats) for cats in self.categories.values())
        result = np.zeros((data.shape[0], total_categories))
        
        current_pos = 0
        for col in range(data.shape[1]):
            for i, val in enumerate(data[:, col]):
                if val in self.categories[col]:
                    result[i, current_pos + self.categories[col][val]] = 1
            current_pos += len(self.categories[col])
            
        return result

class SimpleLabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.mapping = None
        
    def fit_transform(self, labels):
        unique_labels = sorted(set(labels))
        self.classes_ = unique_labels
        self.mapping = {label: i for i, label in enumerate(unique_labels)}
        return np.array([self.mapping[label] for label in labels])
    
    def transform(self, labels):
        return np.array([self.mapping[label] for label in labels])
    
    def inverse_transform(self, indices):
        reverse_mapping = {i: label for label, i in self.mapping.items()}
        return np.array([reverse_mapping[i] for i in indices])