import json
import boto3
import pandas as pd
import numpy as np
from io import StringIO
import os
import sys
import logging

# Add the path to our custom models
sys.path.append('/opt/python')
from models.traditional import (
    SimpleDecisionTree,
    SimpleKNN,
    SimpleRandomForest,
    SimpleLogisticRegression,
    SimpleKMeans,
    SimpleDBSCAN,
    SimpleAgglomerativeClustering,
    SimpleXGBoost
)
from models.preprocessing import (
    SimpleStandardScaler,
    SimpleOneHotEncoder,
    SimpleLabelEncoder
)

# Environment variables
READ_LAMBDA_ENDPOINT = os.getenv('READ_LAMBDA_ENDPOINT', 'data_loader')
lambda_client = boto3.client('lambda')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def handler(event, context):
    try:
        # Parse the event body
        body = json.loads(event['body']) if 'body' in event else event
        filename = body.get('filename')
        algorithm = body.get('algorithm')
        train_columns = body.get('trainColumns')
        target_column = body.get('targetColumn')

         # Drop date column if necessary
        if 'TransactionDate' in train_columns:
            train_columns.remove('TransactionDate')
        if 'TransactionDate' == target_column:
            return _response(400, "TransactionDate cannot be the target column.")

         # Handle unsupervised algorithms
        unsupervised_algorithms = ['K-Means', 'DBSCAN', 'Agglomerative Clustering']
        target_column = target_column if algorithm not in unsupervised_algorithms else 'cluster'

        if not filename:
            return _response(400, "Missing required field: 'filename'.")
        if not algorithm:
            return _response(400, "Missing required field: 'algorithm'.")
        if not train_columns:
            return _response(400, "Missing required field: 'trainColumns'.")

        if algorithm not in unsupervised_algorithms:
            if target_column is None or target_column == '':
                return _response(400, "A target column must be selected for supervised algorithms.")
            # fix for target column validation
            if target_column not in df.columns:
                return _response(400, f"Target column '{target_column}' not found in the dataset.")

        # Invoke the data_loader Lambda function to fetch the CSV data
        invoke_response = lambda_client.invoke(
            FunctionName=READ_LAMBDA_ENDPOINT,
            InvocationType='RequestResponse',
            Payload=json.dumps({'filename': filename})
        )

        # Parse the response from data_loader
        response_payload = json.loads(invoke_response['Payload'].read().decode('utf-8'))
        if response_payload.get('statusCode') != 200:
            return _response(response_payload.get('statusCode', 500), response_payload.get('body', 'Error occurred'))

        # Extract the CSV data
        csv_data = json.loads(response_payload['body']).get('csv')
        if not csv_data:
            return _response(500, "Error: 'csv' key not found in response")

        # Convert the CSV data to a DataFrame
        df = pd.read_csv(StringIO(csv_data))
        df = df.where(pd.notnull(df), None)  # Replace NaN with None

        # Define model mapping
        model_mapping = {
            'Decision Tree': lambda: SimpleDecisionTree(max_depth=5),
            'Decision Tree Regressor': lambda: SimpleDecisionTree(max_depth=5),
            'K-Nearest Neighbors': lambda: SimpleKNN(k=5),
            'K-Nearest Neighbors Regressor': lambda: SimpleKNN(k=5),
            'Random Forest': lambda: SimpleRandomForest(n_trees=10, max_depth=5),
            'Random Forest Regressor': lambda: SimpleRandomForest(n_trees=10, max_depth=5),
            'Logistic Regression': lambda: SimpleLogisticRegression(learning_rate=0.01, max_iter=1000),
            'K-Means': lambda: SimpleKMeans(n_clusters=5),
            'DBSCAN': lambda: SimpleDBSCAN(eps=0.5, min_samples=5),
            'Agglomerative Clustering': lambda: SimpleAgglomerativeClustering(n_clusters=5),
            'XGBoost': lambda: SimpleXGBoost(n_estimators=100, learning_rate=0.1, max_depth=3)
        }

        if algorithm not in model_mapping:
            return _response(400, "Invalid algorithm.")

        model = model_mapping[algorithm]()

        # Splitting the data
        X = df[train_columns].drop(columns=[target_column], errors='ignore')
        y = df[target_column] if target_column in df.columns else None

        # Preprocessing
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

        # Initialize preprocessors
        scaler = SimpleStandardScaler()
        encoder = SimpleOneHotEncoder()
        label_encoder = SimpleLabelEncoder()

        # Preprocess numerical features
        if numerical_cols:
            X_num = scaler.fit_transform(X[numerical_cols].values)
        else:
            X_num = np.array([])

        # Preprocess categorical features
        if categorical_cols:
            X_cat = encoder.fit_transform(X[categorical_cols])
        else:
            X_cat = np.array([])

        # Combine preprocessed features
        if len(X_num) > 0 and len(X_cat) > 0:
            X_processed = np.hstack([X_num, X_cat])
        elif len(X_num) > 0:
            X_processed = X_num
        else:
            X_processed = X_cat

        # Convert target to numerical if needed
        if y is not None and algorithm not in unsupervised_algorithms:
            y = label_encoder.fit_transform(y)

        # Split data for supervised learning
        if algorithm not in unsupervised_algorithms:
            # Simple train-test split (80-20)
            n_samples = len(X_processed)
            indices = np.random.permutation(n_samples)
            split_idx = int(0.8 * n_samples)
            train_idx, test_idx = indices[:split_idx], indices[split_idx:]
            
            X_train, X_test = X_processed[train_idx], X_processed[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            X_train, X_test = X_processed, None
            y_train, y_test = None, None

        # Fit the model
        if algorithm in unsupervised_algorithms:
            model.fit(X_train)

            if algorithm == 'Agglomerative':
                predictions = model.labels_
            else:
                predictions = model.predict(X_train)

            df[target_column] = predictions
            results = {"message": f"{algorithm} Clustering completed"}
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate metrics
            if 'Regressor' in algorithm:
                mse = np.mean((y_test - predictions) ** 2)
                r2 = 1 - mse / np.var(y_test)
                results = {'mse': mse, 'r2': r2}
            else:
                accuracy = np.mean(y_test == predictions)
                # Simple F1 calculation
                tp = np.sum((y_test == 1) & (predictions == 1))
                fp = np.sum((y_test == 0) & (predictions == 1))
                fn = np.sum((y_test == 1) & (predictions == 0))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                results = {'accuracy': accuracy, 'f1_score': f1}

            # Update predictions in dataframe
            df.loc[test_idx, target_column] = predictions

        # Return the results
        return _response(200, {
            'dataframe': df.head(1000).to_dict(orient='records'),
            'model': model.__class__.__name__,
            'results': results,
            'predictions': df[target_column].tolist(),
            'size': len(df),
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return _response(500, f"Error: {str(e)}")


def _response(status_code, body):
     return {
        "statusCode": status_code,
         "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "POST,OPTIONS"
        },
        "body": json.dumps({"error": body}) if status_code != 200 else json.dumps(body)
    }