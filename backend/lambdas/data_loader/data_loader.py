import json
import boto3
import pandas as pd
from io import StringIO
import os

S3_BUCKET = os.getenv('S3_BUCKET', 'j-ml-playground-data')

def handler(event, context):
    try:
        
        if 'body' in event:
            body = event['body']
            if isinstance(body, str):
                body = json.loads(body)  
            elif isinstance(body, dict):
                pass  
            else:
                return _response(400, 'Invalid body format.')
        else:
            body = event
        filename = body.get('filename')
        if not filename:
            return _response(400, 'Filename is required.')

        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=S3_BUCKET, Key=filename)
        csv_string = obj['Body'].read().decode('utf-8')

        df = pd.read_csv(StringIO(csv_string))
        df = df.where(pd.notnull(df), None)
        preview = df.head(100).to_dict(orient='records')

        return _response(200, preview)

    except Exception as e:
        return _response(500, f'Error: {str(e)}')


def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # for CORS
        },
        "body": body if isinstance(body, str) else json.dumps(body) 
}

