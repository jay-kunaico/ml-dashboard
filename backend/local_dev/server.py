from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from pydantic import BaseModel, ValidationError, Field
import boto3
import pandas as pd
import numpy as np
import os
import logging
from config import ENV, DATA_SOURCE
from io import StringIO
import time

from models.traditional import SimpleLogisticRegression,SimpleDecisionTree, SimpleKNN, SimpleRandomForest,SimpleKMeans,SimpleDBSCAN,SimpleAgglomerativeClustering,SimpleXGBoost
from models.preprocessing import SimpleStandardScaler, SimpleOneHotEncoder, SimpleLabelEncoder

app = Flask(__name__)
CORS(app, resources={r"/*": {
         "origins": "*",
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"]
     }}, 
     supports_credentials=True, max_age=600)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Data(BaseModel):
    filename: str

class Algorithm(BaseModel):
    filename: str   
    algorithm: str
    trainColumns: list[str]
    targetColumn: str   

def read_data(filename):
    if ENV == 'cloud':
        s3 = boto3.client('s3')
        try:
            obj = s3.get_object(Bucket=DATA_SOURCE, Key=filename)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            df = df.where(pd.notnull(df), None)
            return df
        except Exception as e:
            raise ValueError(f"Error reading file from S3: {str(e)}")
    else:
        file_path = os.path.join(DATA_SOURCE, filename)
        logging.debug(f"Reading file from local path: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{filename}' not found")
        try:
            df = pd.read_csv(file_path)
            df = df.where(pd.notnull(df), None)
            return df
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

@app.route('/run-preview', methods=['POST'])
def load_preview():
    try:
        data = Data.model_validate(request.json)
        df = read_data(data.filename)
        result = df.head(100).to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    model_mapping = {
            'Logistic Regression': lambda: SimpleLogisticRegression(),
            'Decision Tree': lambda: SimpleDecisionTree(max_depth=3, min_samples_split=20),
            'Decision Tree Regressor': lambda: SimpleDecisionTree(max_depth=3, min_samples_split=20),
            'K-Nearest Neighbors': lambda: SimpleKNN(k=5),
            'K-Nearest Neighbors Regressor': lambda: SimpleKNN(k=5),
            'Random Forest': lambda: SimpleRandomForest(n_trees=10, max_depth=5),
            'Random Forest Regressor': lambda: SimpleRandomForest(n_trees=10, max_depth=5),
            'K-Means': lambda: SimpleKMeans(n_clusters=5),
            'DBSCAN': lambda: SimpleDBSCAN(eps=0.5, min_samples=5),
            'Agglomerative Clustering': lambda: SimpleAgglomerativeClustering(n_clusters=5),
            'XGBoost': lambda: SimpleXGBoost(n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, reg_alpha=0.0)
        }
    clustering_algorithms = ['K-Means', 'DBSCAN', 'Agglomerative Clustering']
    try:
        start_time = time.time()
        data = Algorithm.model_validate(request.json)
        
        df = read_data(data.filename)
		
        if data.algorithm not in model_mapping:
            return jsonify({"error": "Invalid algorithm"}), 400
        
        if data.algorithm not in clustering_algorithms:
        # Supervised: targetColumn must be provided and not empty
            if not data.targetColumn or data.targetColumn.strip() == "":
                return jsonify({"error": "A target column must be selected for supervised models."}), 400
            else:
                # Unsupervised: ignore targetColumn or set to None
                data.targetColumn = None

        # Prepare data
        if data.algorithm in clustering_algorithms:
            X = df[data.trainColumns]
        else:
            if data.targetColumn in data.trainColumns:
                data.trainColumns.remove(data.targetColumn)
            X = df[data.trainColumns]
            y = df[data.targetColumn]

        # Drop date column if necessary 
        if 'TransactionDate' in data.trainColumns:
            data.trainColumns.remove('TransactionDate')
        if 'TransactionDate' == data.targetColumn:
            return jsonify({"error": "TransactionDate cannot be the target column"}), 400

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

        # Preprocess data
        num_scaler = SimpleStandardScaler()
        cat_encoder = SimpleOneHotEncoder()
        
        # Process numerical and categorical data separately
        if numerical_cols:
            X_num = num_scaler.fit_transform(X[numerical_cols].values)
        else:
            X_num = np.array([]).reshape(len(X), 0)
            
        if categorical_cols:
            X_cat = cat_encoder.fit_transform(X[categorical_cols])
        else:
            X_cat = np.array([]).reshape(len(X), 0)
        
        # Combine processed features
        X_processed = np.hstack([X_num, X_cat])

        model = model_mapping[data.algorithm]()
        
        if data.algorithm in clustering_algorithms:
            model.fit(X_processed)
            predictions = model.predict(X_processed)
            
            # Calculate clustering-specific metrics
            n_clusters = len(np.unique(predictions[predictions != -1]))  # Don't count noise points (-1)
            unique_clusters, counts = np.unique(predictions, return_counts=True)
            cluster_sizes = {int(cluster): int(count) for cluster, count in zip(unique_clusters, counts)}
            results = {
                "message": f"{data.algorithm} completed",
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes
            }
            
            # Update the dataframe with cluster assignments
            df['cluster'] = predictions
            target_column = 'cluster'  # Use this for the response
            
        else:
            # Handle supervised learning
            preprocessing_time = time.time() - start_time
            
            if 'Regressor' not in data.algorithm:
                label_encoder = SimpleLabelEncoder()
                y_processed = label_encoder.fit_transform(y)
            else:
                y_processed = y.values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42
            )

            # Train model with timing
            training_start = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - training_start
            
            # Make predictions with timing
            prediction_start = time.time()
            predictions = model.predict(X_test)
            prediction_time = time.time() - prediction_start

            # Calculate metrics
            if 'Regressor' in data.algorithm:
                mse = np.mean((y_test - predictions) ** 2)
                r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
                results = {'mse': float(mse), 'r2': float(r2)}
            else:
                accuracy = np.mean(predictions == y_test)
                # Simple f1 score implementation for binary classification
                if len(set(y_test)) == 2:
                    true_pos = np.sum((predictions == 1) & (y_test == 1))
                    false_pos = np.sum((predictions == 1) & (y_test == 0))
                    false_neg = np.sum((predictions == 0) & (y_test == 1))
                    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    f1 = accuracy  # Simplified for multiclass
                results = {'accuracy': float(accuracy), 'f1_score': float(f1)}

            # Update predictions in dataframe
            test_indices = np.random.permutation(len(df))[:len(y_test)]
            if 'Regressor' not in data.algorithm:
                df.loc[test_indices, data.targetColumn] = label_encoder.inverse_transform(predictions)
            else:
                df.loc[test_indices, data.targetColumn] = predictions
            target_column = data.targetColumn

        total_time = time.time() - start_time

        # Add timing information to results
        results['processing_time'] = float(total_time)

        return jsonify({
            'model': data.algorithm,
            'results': results,
            'predictions': df[target_column].tolist(),
            'dataframe': df.head(1000).to_dict(orient='records'),
            'size': len(df),
        })

    except ValidationError as ve:
        return jsonify({"error": str(ve)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-preview', methods=['OPTIONS'])
def handle_options_run_preview():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/run_algorithm', methods=['OPTIONS'])
def handle_options_run_algorithm():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

if __name__ == '__main__':
    logging.getLogger('flask_cors').level = logging.DEBUG
    app.run(debug=True)