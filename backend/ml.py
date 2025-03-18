from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import os
import logging

app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

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
    logging.debug("body: %s", request.get_data())
    logging.debug(f"Data: {request.data}")

# Path to data
CSV_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
df = None

@app.route('/load_data', methods=['POST'])
def read_csv():
    global df
    data = request.json
    filename = data.get('filename')
    mode = data.get('mode', 'preview')

    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    file_path = os.path.join(CSV_FOLDER, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": f"File '{filename}' not found"}), 404

    try:
        df = pd.read_csv(file_path)
        df = df.where(pd.notnull(df), None)
        if mode == 'preview':
            return jsonify(df.head(10).to_dict(orient='records'))
        elif mode == 'full':
            return jsonify(df.head(1000).to_dict(orient='records'))
        result = df.head(1000).to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    global df
    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    data = request.json
    logging.info(f"Received data: {data}")
    algorithm = data.get('algorithm')
    train_columns = data.get('trainColumns')
    target_column = data.get('targetColumn')
    test_size = len(df)

    if not algorithm:
        return jsonify({"error": "Algorithm is required"}), 400
    if not train_columns:
        return jsonify({"error": "Train columns are required"}), 400
      # Handle case where target_column is not provided
    if not target_column:
        # Dynamically create a target column
        target_column = 'prediction'
        df[target_column] = 0  # Initialize the column with None

    logging.info("target_column: %s", target_column)
    logging.info("train_columns: %s", train_columns)

    # Drop date column if necessary
    if 'TransactionDate' in train_columns:
        train_columns.remove('TransactionDate')
    if 'TransactionDate' == target_column:
        return jsonify({"error": "TransactionDate cannot be the target column"}), 400

    try:
        # Splitting the data
        X = df[train_columns].values
        y = df[target_column].values

        X = pd.DataFrame(X, columns=train_columns)

        if y is None or all(pd.isnull(y)):
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            y_train, y_test = None, None
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

    model = None

    # Initialize the model
    if algorithm == 'Linear Regression':
        model = LinearRegression()
    elif algorithm == 'Decision Tree':
        model = DecisionTreeRegressor()
    elif algorithm == 'K-Nearest Neighbors':
        model = KNeighborsRegressor(n_neighbors=5)
    elif algorithm == 'Random Forest':
        model = RandomForestRegressor()
    elif algorithm == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif algorithm == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        return jsonify({"error": "Invalid algorithm"}), 400

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Fit the model
    if y_train is not None and not all(pd.isnull(y_train)):
        pipeline.fit(X_train, y_train)

        # Make predictions
        predictions = pipeline.predict(X_test)

        df.loc[X_test.index, target_column] = predictions

        # Calculate metrics
        if algorithm in [
            'Linear Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Random Forest'
        ]:
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            results = {'mse': mse, 'r2': r2}
        else:
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            results = {'accuracy': accuracy, 'f1_score': f1}
    else:
       pipeline.fit(X)
       predictions = pipeline.predict(X)
       df[target_column] = predictions
    #    results = {"message": "Predictions completed without a target column"}

    return jsonify({
        'model': model.__class__.__name__,
        'results': results,
        'predictions': df[target_column].tolist(),
        'dataframe': df.to_dict(orient='records'),
        'size': test_size,
    })

if __name__ == '__main__':
    app.run(debug=True)