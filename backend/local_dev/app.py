from lambdas.data_loader import read_data
import json
import logging



# from flask import Flask, request, jsonify, make_response
# from flask_cors import CORS
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler	
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor
# # import numpy as np
# import os

# import logging

# app = Flask(__name__)
# # CORS(app)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # Path to data
# CSV_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
# df=None

# @app.before_request
# def before_request():
#     if(request.method == 'OPTIONS'):
#         response = make_response()
#         response.headers['Access-Control-Allow-Origin'] = '*'
#         response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
#         response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
#         return response

# @app.route('/load_data', methods=['POST'])
# def read_csv():
#     global df
#     data = request.json
#     filename = data.get('filename')
#     mode = data.get('mode', 'preview')

#     if not filename:
#         return jsonify({"error": "Filename is required"}), 400

#     file_path = os.path.join(CSV_FOLDER, filename)
#     logging.info(f"Reading file '{filename}' from '{file_path}'")

#     if not os.path.exists(file_path):
#         return jsonify({"error": f"File '{filename}' not found"}), 404

#     try:
#         df = pd.read_csv(file_path)
#         if mode == 'preview':
#             return jsonify(df.head(10).to_dict(orient='records'))
#         elif mode == 'full':
#             return jsonify(df.head(1000).to_dict(orient='records'))
#         limited_df = df.head(1000)
#         result = limited_df.to_dict(orient='records')
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/run_algorithm', methods=['POST'])
# def run_algorithm():
#     global df
#     if df is None:
#         return jsonify({"error": "No data loaded"}), 400
#     data = request.json
#     algorithm = data.get('algorithm')
#     train_columns = data.get('trainColumns')
#     target_column = data.get('targetColumn')
#     test_size = len(df)
#     model = None
    
#     if not algorithm:
#         return jsonify({"error": "Algorithm is required"}), 400
#     if not train_columns:
#         return jsonify({"error": "Train columns are required"}), 400
#     if not target_column:
#         return jsonify({"error": "Target column is required"}), 400
    
#     if algorithm == 'Linear Regression':
#         model = LinearRegression()
#     elif algorithm == 'Decision Tree':
#         model = DecisionTreeRegressor()
#     elif algorithm == 'K-Nearest Neighbors':
#         model = KNeighborsRegressor()
#     else:
#         return jsonify({"error": "Invalid algorithm"}), 400


#     # Splitting the data into training and testing data
#     try:
#         X_train = df[train_columns].values
#         y_train = df[target_column].values
#         X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)	
        
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
#     model.fit(X_train_scaled, y_train)
#     predictions = model.predict(X_test_scaled)
#     mse = mean_squared_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
 
        
#     return jsonify({
# 		'model': model.__class__.__name__,
# 		'mse': mse,
# 		'r2': r2,
# 		'predictions': predictions.tolist(),
#         'size': test_size
# 	})
    

# if __name__ == '__main__':
#     app.run(debug=True)
