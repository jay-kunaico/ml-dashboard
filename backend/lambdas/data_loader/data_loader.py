import json
import boto3
import os

S3_BUCKET = os.getenv('S3_BUCKET', 'j-ml-playground-data')

def handler(event, context):
    try:
        # Parse the event body
        body = json.loads(event['body']) if 'body' in event else event
        filename = body.get('filename')
        if not filename:
            return _response(400, 'Filename is required.')

        # Fetch the file from S3
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=S3_BUCKET, Key=filename)
        csv_string = obj['Body'].read().decode('utf-8')

        # Return the raw CSV string
        return _response(200, {"csv": csv_string})

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