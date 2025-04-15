
from dotenv import load_dotenv
import os
load_dotenv()

ENV = os.getenv('APP_ENV', 'local')

if ENV == 'cloud':
	DATA_SOURCE = os.getenv('S3_BUCKET_NAME')
else:
	DATA_SOURCE = os.getenv('DATA_SOURCE', './data')