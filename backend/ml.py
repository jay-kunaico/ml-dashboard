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

#Pydantic Models
class Data(BaseModel):
    filename: str

class Algorithm(BaseModel):
    filename: str   
    algorithm: str
    trainColumns: list[str]
    targetColumn: str   

# Attempts to address CORS issue in preflight
# flask-cors does this for us
# @app.before_request
# def log_request_info():
#     if(request.method == 'OPTIONS'):
#         response = make_response()
#         response.headers['Access-Control-Allow-Origin'] = '*'
#         response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
#         response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
#         return response
#     logging.info(f"Request: {request.method} {request.url}")
#     logging.debug(f"Headers: {request.headers}")

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
    try:
        data = Data.model_validate(request.json)
        df = read_csv_internal(data.filename)
        # Only return 100 rows to preview data
        result = df.head(100).to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
      

@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    try:
        data = Algorithm.model_validate(request.json)
        df = read_csv_internal(data.filename)

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

        if data.algorithm not in model_mapping:
            return jsonify({"error": "Invalid algorithm"}), 400

        model = model_mapping[data.algorithm]()

        unsupervised_algorithms = ['K-Means', 'DBSCAN', 'Agglomerative Clustering', 'Birch']
        target_column = data.targetColumn if data.algorithm not in unsupervised_algorithms else 'cluster'
        # df[target_column] = 0  # Initialize the column with 0.  this causes an error 

        # Drop date column if necessary
        if 'TransactionDate' in data.trainColumns:
            data.trainColumns.remove('TransactionDate')
        if 'TransactionDate' == target_column:
            return jsonify({"error": "TransactionDate cannot be the target column"}), 400

        # Splitting the data standard step in Machine Learning
        X = df[data.trainColumns].drop(columns=[target_column], errors='ignore')
        y = df[target_column] if target_column in df.columns else None

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

       
        if data.algorithm in unsupervised_algorithms:
            y_train, y_test = None, None
        else:   
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if y_train is not None and len(y_train.unique()) < 2:
            return jsonify({"error": "The training data must contain at least two classes."}), 400
        
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

        # Fit the model / Training and Prediction
        if data.algorithm in unsupervised_algorithms:
            pipeline.fit(X)
            predictions = pipeline.predict(X) if hasattr(model, 'predict') else pipeline.fit_predict(X)
            df[target_column] = predictions
            results = {"message": f"{data.algorithm} Clustering completed"}
        else:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            df.loc[X_test.index, target_column] = predictions

            #  metrics
            if 'Regressor' in data.algorithm:
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                results = {'mse': mse, 'r2': r2}
            else:
                accuracy = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions, average='weighted')
                results = {'accuracy': accuracy, 'f1_score': f1}

        # Return info to the user.  Limit the number of rows returned to 1000 to avoid overloading the server.  Should add pagination or infinite scrolling. TODO.
        return jsonify({
            'model': model.__class__.__name__,
            'results': results,
            'predictions': df[target_column].tolist(),
            'dataframe': df.head(1000).to_dict(orient='records'),
            'size': len(df),
        })

    except ValidationError as ve:
        logging.error(f"ValidationError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        return jsonify({"error": str(e)}), 404
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Error: {e}, {len(df)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)