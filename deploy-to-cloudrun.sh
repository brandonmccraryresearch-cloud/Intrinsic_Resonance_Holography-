#!/bin/bash
# Quick deployment script for IRH to Google Cloud Run
# THEORETICAL FOUNDATION: IRH v21.1 Manuscript
#
# Usage:
#   ./deploy-to-cloudrun.sh PROJECT_ID [REGION]
#
# Example:
#   ./deploy-to-cloudrun.sh my-gcp-project us-central1

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ -z "$1" ]; then
    echo -e "${RED}Error: PROJECT_ID is required${NC}"
    echo "Usage: $0 PROJECT_ID [REGION]"
    echo "Example: $0 my-gcp-project us-central1"
    exit 1
fi

PROJECT_ID=$1
REGION=${2:-us-central1}
SERVICE_NAME="irh-backend"

echo -e "${GREEN}=== IRH v21.1 Cloud Run Deployment ===${NC}"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Name: $SERVICE_NAME"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo -e "${YELLOW}Setting GCP project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy using Cloud Build
echo -e "${YELLOW}Building and deploying to Cloud Run...${NC}"
gcloud builds submit --config cloudbuild.yaml

# Get service URL
echo -e "${GREEN}=== Deployment Complete! ===${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo -e "${GREEN}Service URL: $SERVICE_URL${NC}"
echo ""
echo "Test your deployment:"
echo "  curl $SERVICE_URL/health"
echo "  curl $SERVICE_URL/api/v1/fixed-point"
echo ""
echo "API Documentation:"
echo "  $SERVICE_URL/docs"
echo ""
echo -e "${GREEN}✓ Deployment successful!${NC}"
