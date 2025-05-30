#!/bin/bash

# Create a temporary directory for packaging
mkdir -p packages/models

# Copy the necessary files
cp run_algorithm.py packages/
cp ../../local_dev/models/traditional.py packages/models/
cp ../../local_dev/models/preprocessing.py packages/models/

# Create the deployment package
cd packages
zip -r ../deployment_package.zip .
cd ..

# Clean up
rm -rf packages

echo "Deployment package created: deployment_package.zip" 