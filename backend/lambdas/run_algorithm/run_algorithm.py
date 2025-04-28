import json
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from io import StringIO
import os
import logging

# Environment variables
READ_LAMBDA_ENDPOINT = os.getenv('READ_LAMBDA_ENDPOINT', 'data_loader')
lambda_client = boto3.client('lambda')

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def handler(event, context):
    try:
        # Parse the event body
        body = json.loads(event['body']) if 'body' in event else event
        filename = body.get('filename')
        algorithm = body.get('algorithm')
        train_columns = body.get('trainColumns')
        target_column = body.get('targetColumn')

        if not filename or not algorithm or not train_columns or not target_column:
            return _response(400, "Missing required fields: 'filename', 'algorithm', 'trainColumns', or 'targetColumn'.")

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
            'Decision Tree': DecisionTreeClassifier,
            'Decision Tree Regressor': lambda: DecisionTreeRegressor(max_depth=5),
            'K-Nearest Neighbors': lambda: KNeighborsClassifier(n_neighbors=5),
            'K-Nearest Neighbors Regressor': lambda: KNeighborsRegressor(n_neighbors=5),
            'Random Forest': RandomForestClassifier,
            'Random Forest Regressor': RandomForestRegressor,
            'Logistic Regression': lambda: LogisticRegression(max_iter=1000),
            'XGBoost': lambda: XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'K-Means': lambda: KMeans(n_clusters=5, random_state=42),
            'Agglomerative Clustering': lambda: AgglomerativeClustering(n_clusters=5),
            'DBSCAN': lambda: DBSCAN(eps=0.5, min_samples=5),
            'Birch': lambda: Birch(n_clusters=5)
        }

        if algorithm not in model_mapping:
            return _response(400, "Invalid algorithm.")

        model = model_mapping[algorithm]()

        # Handle unsupervised algorithms
        unsupervised_algorithms = ['K-Means', 'DBSCAN', 'Agglomerative Clustering', 'Birch']
        target_column = target_column if algorithm not in unsupervised_algorithms else 'cluster'

        # Drop date column if necessary
        if 'TransactionDate' in train_columns:
            train_columns.remove('TransactionDate')
        if 'TransactionDate' == target_column:
            return _response(400, "TransactionDate cannot be the target column.")

        # Splitting the data
        X = df[train_columns].drop(columns=[target_column], errors='ignore')
        y = df[target_column] if target_column in df.columns else None

        if algorithm in unsupervised_algorithms:
            y_train, y_test = None, None
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if y_train is not None and len(y_train.unique()) < 2:
            return _response(400, "The training data must contain at least two classes.")

        # Preprocessing
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ])

        # Create a pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Fit the model
        if algorithm in unsupervised_algorithms:
            pipeline.fit(X)
            predictions = pipeline.predict(X) if hasattr(model, 'predict') else pipeline.fit_predict(X)
            df[target_column] = predictions
            results = {"message": f"{algorithm} Clustering completed"}
        else:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            df.loc[X_test.index, target_column] = predictions

            # Metrics
            if 'Regressor' in algorithm:
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                results = {'mse': mse, 'r2': r2}
            else:
                accuracy = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions, average='weighted')
                results = {'accuracy': accuracy, 'f1_score': f1}

        # Return the results
        return _response(200, {
            'model': model.__class__.__name__,
            'results': results,
            'predictions': df[target_column].tolist(),
            'dataframe': df.head(1000).to_dict(orient='records'),
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
            "Access-Control-Allow-Origin": "*",  # for CORS
        },
        "body": json.dumps(body) if not isinstance(body, str) else body
    }