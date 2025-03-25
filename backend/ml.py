from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from pydantic import BaseModel, ValidationError, Field
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN,Birch
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import os
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True, max_age=600)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



@app.before_request
def log_request_info():
    if(request.method == 'OPTIONS'):
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    logging.info(f"Request: {request.method} {request.url}")
    logging.debug(f"Headers: {request.headers}")

# Path to data
CSV_FOLDER = os.path.join(os.path.dirname(__file__), 'data')

def read_csv_internal(filename):
    file_path = os.path.join(CSV_FOLDER, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{filename}' not found")

    try:
        df = pd.read_csv(file_path)
        df = df.where(pd.notnull(df), None)  # Replace NaN with None
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")   

@app.route('/load_data', methods=['POST'])
def load_preview():
    data = request.json
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    file_path = os.path.join(CSV_FOLDER, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": f"File '{filename}' not found"}), 404

    try:
        df = pd.read_csv(file_path)
        df = df.where(pd.notnull(df), None)
        result = df.head(100).to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    data = request.json
    filename = data.get('filename')  
    algorithm = data.get('algorithm')
    train_columns = data.get('trainColumns')
    target_column = data.get('targetColumn')

    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    if not algorithm:
        return jsonify({"error": "Algorithm is required"}), 400
    if not train_columns:
        return jsonify({"error": "Train columns are required"}), 400

    try:
        df = read_csv_internal(filename)

        if not target_column:
            target_column = 'cluster' if algorithm in ["K-Means", "DBSCAN", "Agglomerative Clustering", "Birch"] else 'prediction'
            df[target_column] = 0  # Initialize the column with 0

        # Drop date column if necessary
        if 'TransactionDate' in train_columns:
            train_columns.remove('TransactionDate')
        if 'TransactionDate' == target_column:
            return jsonify({"error": "TransactionDate cannot be the target column"}), 400

        # Splitting the data
        X = df[train_columns]
        y = df[target_column] if target_column in df.columns else None

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

        unsupervised_algorithms = ['K-Means', 'DBSCAN', 'Agglomerative Clustering', 'Birch']
        if algorithm in unsupervised_algorithms:
            y_train, y_test = None, None
        else:   
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ])

        model = None

        # Initialize the model
        if algorithm == 'Decision Tree':
            model = DecisionTreeClassifier()
        elif algorithm == "Decision Tree Regressor":
            model = DecisionTreeRegressor(max_depth=5)
        elif algorithm == 'K-Nearest Neighbors':
            model = KNeighborsClassifier(n_neighbors=5)
        elif algorithm == 'K-Nearest Neighbors Regressor':
            model = KNeighborsRegressor(n_neighbors=5)
        elif algorithm == 'Random Forest':
            model = RandomForestClassifier()
        elif algorithm == 'Random Forest Regressor':
            model = RandomForestRegressor()
        elif algorithm == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000)
        elif algorithm == 'XGBoost':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        elif algorithm == 'K-Means':
            model = KMeans(n_clusters=5, random_state=42)
        elif algorithm == 'Agglomerative Clustering':
            model = AgglomerativeClustering(n_clusters=5)
        elif algorithm == 'DBSCAN':
            model = DBSCAN(eps=0.5, min_samples=5)
        elif algorithm == 'Birch':
            model = Birch(n_clusters=5)
        else:
            return jsonify({"error": "Invalid algorithm"}), 400

        # Create pipeline
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

            # Calculate metrics
            if 'Regressor' in algorithm:
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                results = {'mse': mse, 'r2': r2}
            else:
                accuracy = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions, average='weighted')
                results = {'accuracy': accuracy, 'f1_score': f1}

        return jsonify({
            'model': model.__class__.__name__,
            'results': results,
            'predictions': df[target_column].tolist(),
            'dataframe': df.head(1000).to_dict(orient='records'),
            'size': len(df),
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)