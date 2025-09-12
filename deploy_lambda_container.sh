#!/bin/bash

# PLUG-AND-PLAY CONFIGURATION (no edits needed)
LAMBDA_NAME="OmegaSingularityInference"
AWS_REGION="us-east-2"
ACCOUNT_ID="970982543175"
ECR_REPO="omega-singularity-lambda"
IMAGE_TAG="latest"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

# 1. Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl enable --now docker
    sudo usermod -aG docker $USER
    echo "Docker installed. Please log out and back in, or run: newgrp docker"
    exit 1
fi

# 2. Authenticate Docker to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# 3. Create ECR repository if it doesn't exist
echo "Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION > /dev/null 2>&1 || \
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

# 4. Build Docker image
echo "Building Docker image..."
docker build -t $ECR_REPO .

# 5. Tag Docker image
echo "Tagging Docker image..."
docker tag $ECR_REPO:latest $IMAGE_URI

# 6. Push Docker image to ECR
echo "Pushing Docker image to ECR..."
docker push $IMAGE_URI

# 7. Update Lambda function to use the new container image
echo "Updating Lambda function to use new container image..."
aws lambda update-function-code \
  --function-name $LAMBDA_NAME \
  --image-uri $IMAGE_URI \
  --region $AWS_REGION

echo "Deployment complete!"
