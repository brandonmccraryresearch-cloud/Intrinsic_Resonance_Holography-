# Google Cloud Run Deployment Guide - IRH v21.1

This guide covers deploying the Intrinsic Resonance Holography (IRH) computational framework backend to Google Cloud Run.

## Overview

Google Cloud Run provides a fully managed serverless platform for containerized applications. The IRH backend API has been optimized for Cloud Run with:

- ✅ Dynamic port binding (via `PORT` environment variable)
- ✅ Stateless architecture
- ✅ Fast cold starts (optimized image size)
- ✅ Health check endpoints
- ✅ Non-root user execution
- ✅ Auto-scaling (0 to N instances)

## Prerequisites

### 1. Install Google Cloud SDK

```bash
# macOS
brew install google-cloud-sdk

# Ubuntu/Debian
sudo apt-get install google-cloud-sdk

# Or use the installer
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### 2. Initialize gcloud

```bash
gcloud init
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Enable Required APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Deployment Methods

### Method 1: Quick Deploy (Automated with Cloud Build)

This is the **recommended** approach for production deployments.

```bash
# From repository root
gcloud builds submit --config cloudbuild.yaml

# Wait for build to complete (5-10 minutes)
# Your service will be automatically deployed to Cloud Run
```

After deployment completes, you'll see output like:
```
Service [irh-backend] revision [irh-backend-00001-abc] has been deployed and is serving 100 percent of traffic.
Service URL: https://irh-backend-abc123-uc.a.run.app
```

### Method 2: Manual Build and Deploy

For more control or testing:

#### Step 1: Build the Docker Image

```bash
# Build locally
docker build -f Dockerfile.cloudrun -t gcr.io/YOUR_PROJECT_ID/irh-backend:latest .

# Or build with Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/irh-backend:latest -f Dockerfile.cloudrun
```

#### Step 2: Deploy to Cloud Run

```bash
gcloud run deploy irh-backend \
  --image gcr.io/YOUR_PROJECT_ID/irh-backend:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 60s \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars PYTHONPATH=/app,LOG_LEVEL=INFO
```

### Method 3: Local Testing First

Test the Docker image locally before deploying:

```bash
# Build image
docker build -f Dockerfile.cloudrun -t irh-backend:local .

# Run locally on port 8080
docker run -p 8080:8080 -e PORT=8080 irh-backend:local

# Test in another terminal
curl http://localhost:8080/health
curl http://localhost:8080/api/v1/fixed-point
```

## Configuration Options

### Environment Variables

Set environment variables during deployment:

```bash
gcloud run deploy irh-backend \
  --set-env-vars "PYTHONPATH=/app,LOG_LEVEL=INFO,ALLOWED_ORIGINS=https://yourdomain.com"
```

### Resource Limits

Adjust memory and CPU based on your needs:

| Use Case | Memory | CPU | Concurrency |
|----------|--------|-----|-------------|
| **Light** (dev/test) | 512Mi | 1 | 40 |
| **Standard** (production) | 2Gi | 2 | 80 |
| **Heavy** (high traffic) | 4Gi | 4 | 100 |

Example:
```bash
gcloud run deploy irh-backend \
  --memory 4Gi \
  --cpu 4 \
  --concurrency 100
```

### Auto-scaling

Configure min/max instances:

```bash
gcloud run deploy irh-backend \
  --min-instances 1 \     # Keep 1 instance warm (avoids cold starts)
  --max-instances 100     # Allow up to 100 instances for high traffic
```

### Timeout

Cloud Run default timeout is 60s. Increase for long-running computations:

```bash
gcloud run deploy irh-backend \
  --timeout 300s  # 5 minutes max
```

**Note**: Maximum timeout is 60 minutes for Cloud Run (2nd gen).

## Security Configuration

### Private Service (Authentication Required)

Deploy with authentication:

```bash
gcloud run deploy irh-backend \
  --no-allow-unauthenticated

# Then grant access to specific service accounts or users
gcloud run services add-iam-policy-binding irh-backend \
  --member="user:someone@example.com" \
  --role="roles/run.invoker"
```

### Custom Domain

Map a custom domain to your service:

```bash
# First, verify domain ownership in Cloud Console
gcloud domains verify example.com

# Map domain to service
gcloud run domain-mappings create \
  --service irh-backend \
  --domain api.example.com \
  --region us-central1
```

### HTTPS

Cloud Run automatically provisions and renews TLS certificates for custom domains.

## Continuous Deployment

### GitHub Integration

Set up automatic deployments on push to main:

```bash
gcloud builds triggers create github \
  --repo-name=Intrinsic_Resonance_Holography- \
  --repo-owner=brandonmccraryresearch-cloud \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml
```

Now every push to `main` triggers a new deployment.

### Manual Trigger

Manually trigger a build:

```bash
gcloud builds submit --config cloudbuild.yaml
```

## Testing the Deployment

### Health Check

```bash
SERVICE_URL=$(gcloud run services describe irh-backend --region us-central1 --format 'value(status.url)')
curl $SERVICE_URL/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "21.0.0",
  "irh_available": true,
  "modules_loaded": ["rg_flow.fixed_points", ...]
}
```

### API Endpoints

Test key endpoints:

```bash
# Fixed Point
curl $SERVICE_URL/api/v1/fixed-point

# Fine-structure constant
curl $SERVICE_URL/api/v1/observables/alpha

# Dark energy
curl $SERVICE_URL/api/v1/observables/dark-energy

# Standard Model gauge group
curl $SERVICE_URL/api/v1/standard-model/gauge-group

# API documentation
open $SERVICE_URL/docs
```

## Monitoring

### View Logs

```bash
# Real-time logs
gcloud run services logs tail irh-backend --region us-central1

# Logs in Cloud Console
gcloud run services describe irh-backend --region us-central1 --format 'value(status.url)' | xargs -I {} open {}
```

### Metrics

View metrics in Cloud Console:
- **Requests**: Request count, latency, errors
- **Container**: CPU, memory, startup time
- **Billable time**: Container instance hours

```bash
# Open Cloud Console metrics
gcloud run services describe irh-backend --region us-central1 --format 'value(status.url)' | sed 's|https://||' | xargs -I {} open "https://console.cloud.google.com/run/detail/us-central1/irh-backend/metrics"
```

## Cost Optimization

### Pricing Model

Cloud Run charges for:
1. **CPU**: Only while processing requests
2. **Memory**: Only while processing requests
3. **Requests**: $0.40 per million requests

### Optimization Tips

1. **Use min-instances=0** for low-traffic services (saves idle costs)
2. **Use min-instances=1+** for services needing fast response (avoids cold starts)
3. **Right-size resources**: Don't over-provision CPU/memory
4. **Enable concurrency**: Handle multiple requests per instance

### Cost Estimate

For moderate usage (10,000 requests/month, 2Gi RAM, 2 CPU):
- ~$5-10/month

For high usage (1M requests/month, 4Gi RAM, 4 CPU, min 2 instances):
- ~$100-200/month

Use the [Cloud Run pricing calculator](https://cloud.google.com/products/calculator) for accurate estimates.

## Troubleshooting

### Image Build Fails

**Problem**: Docker build fails or times out.

**Solution**:
```bash
# Use Cloud Build with high-CPU machine
gcloud builds submit --timeout=30m --machine-type=N1_HIGHCPU_8 --tag gcr.io/PROJECT_ID/irh-backend
```

### Container Fails to Start

**Problem**: Container exits immediately after deployment.

**Solution**: Check logs for errors:
```bash
gcloud run services logs read irh-backend --region us-central1 --limit 50
```

Common issues:
- Missing dependencies → Update `requirements.txt`
- Import errors → Check `PYTHONPATH=/app` is set
- Port binding → Ensure app uses `PORT` environment variable

### Health Check Fails

**Problem**: Service shows as unhealthy.

**Solution**: Test health endpoint locally:
```bash
docker run -p 8080:8080 -e PORT=8080 gcr.io/PROJECT_ID/irh-backend
curl http://localhost:8080/health
```

### Timeout Errors

**Problem**: Requests timeout after 60s.

**Solution**: Increase timeout:
```bash
gcloud run services update irh-backend --timeout 300s --region us-central1
```

### Cold Start Latency

**Problem**: First request after idle is slow (5-10s).

**Solutions**:
1. Keep 1+ instance warm: `--min-instances 1`
2. Optimize image size (already done in Dockerfile.cloudrun)
3. Use Cloud Run 2nd generation for faster cold starts

### Authentication Issues

**Problem**: 403 Forbidden when accessing service.

**Solution**: Check IAM permissions:
```bash
# Make service public
gcloud run services add-iam-policy-binding irh-backend \
  --member="allUsers" \
  --role="roles/run.invoker"

# Or grant specific user access
gcloud run services add-iam-policy-binding irh-backend \
  --member="user:someone@example.com" \
  --role="roles/run.invoker"
```

## Advanced Configuration

### Multiple Environments

Deploy separate dev/staging/production services:

```bash
# Development
gcloud run deploy irh-backend-dev \
  --image gcr.io/PROJECT_ID/irh-backend:dev \
  --set-env-vars ENV=development

# Staging
gcloud run deploy irh-backend-staging \
  --image gcr.io/PROJECT_ID/irh-backend:staging \
  --set-env-vars ENV=staging

# Production
gcloud run deploy irh-backend-prod \
  --image gcr.io/PROJECT_ID/irh-backend:latest \
  --set-env-vars ENV=production \
  --min-instances 2
```

### Traffic Splitting (Blue-Green Deployments)

Deploy new version with gradual rollout:

```bash
# Deploy new version with tag
gcloud run deploy irh-backend \
  --image gcr.io/PROJECT_ID/irh-backend:v2 \
  --tag blue \
  --no-traffic

# Test blue version
curl https://blue---irh-backend-abc123-uc.a.run.app/health

# Shift traffic gradually
gcloud run services update-traffic irh-backend \
  --to-revisions blue=10

# If successful, shift all traffic
gcloud run services update-traffic irh-backend \
  --to-latest
```

### VPC Connector (Private Network Access)

Connect to VPC resources (Cloud SQL, Memorystore, etc.):

```bash
# Create VPC connector
gcloud compute networks vpc-access connectors create irh-connector \
  --network default \
  --region us-central1 \
  --range 10.8.0.0/28

# Deploy with VPC connector
gcloud run deploy irh-backend \
  --vpc-connector irh-connector \
  --vpc-egress all-traffic
```

## API Documentation

Once deployed, access interactive API docs at:
- **Swagger UI**: `https://your-service-url.run.app/docs`
- **ReDoc**: `https://your-service-url.run.app/redoc`

## Support and Resources

- **IRH Repository**: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-
- **Cloud Run Docs**: https://cloud.google.com/run/docs
- **Cloud Build Docs**: https://cloud.google.com/build/docs
- **Pricing Calculator**: https://cloud.google.com/products/calculator

## Quick Reference

```bash
# Build and deploy
gcloud builds submit --config cloudbuild.yaml

# View service URL
gcloud run services describe irh-backend --region us-central1 --format 'value(status.url)'

# View logs
gcloud run services logs tail irh-backend --region us-central1

# Update service
gcloud run services update irh-backend --region us-central1 --memory 4Gi

# Delete service
gcloud run services delete irh-backend --region us-central1
```

---

**Last Updated**: December 2025  
**Version**: IRH v21.1  
**Maintained by**: IRH Computational Framework Team
