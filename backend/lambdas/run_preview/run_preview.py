import json
import boto3
import pandas as pd
from io import StringIO
import os

READ_LAMBDA_ENDPOINT = os.getenv('READ_LAMBDA_ENDPOINT', 'data_loader')
lambda_client = boto3.client('lambda')

def handler(event, context):
    try:
        # Parse the event body
        body = json.loads(event['body']) if 'body' in event else event
        filename = body.get('filename')
        if not filename:
            return _response(400, 'Filename is required.')

        # Invoke the data_loader Lambda function
        invoke_response = lambda_client.invoke(
            FunctionName=READ_LAMBDA_ENDPOINT,
            InvocationType='RequestResponse',
            Payload=json.dumps({'filename': filename})
        )

        # Parse the response from data_loader
        response_payload = json.loads(invoke_response['Payload'].read().decode('utf-8'))
        if response_payload.get('statusCode') != 200:
            return _response(response_payload.get('statusCode', 500), response_payload.get('body', 'Error occurred'))

        # Extract and process the CSV data
        csv_data = json.loads(response_payload['body']).get('csv')
        if not csv_data:
            return _response(500, "Error: 'csv' key not found in response")

        df = pd.read_csv(StringIO(csv_data))
        df = df.where(pd.notnull(df), None)  # Replace NaN with None
        preview = df.head(100).to_dict(orient='records')

        # Return the preview to the frontend
        return _response(200, preview)

    except Exception as e:
        return _response(500, f"Error: {str(e)}")


def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # for CORS
        },
        "body": json.dumps(body)  # Always serialize the body as JSON
    }