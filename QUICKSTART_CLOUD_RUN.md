# Quick Start: Deploy IRH to Google Cloud Run

Get the IRH computational framework running on Google Cloud Run in under 10 minutes.

## Prerequisites

- Google Cloud account ([sign up here](https://cloud.google.com/free))
- gcloud CLI installed ([install guide](https://cloud.google.com/sdk/docs/install))

## Step 1: Set Up Google Cloud

```bash
# Login to Google Cloud
gcloud auth login

# Create a new project (or use existing)
gcloud projects create irh-project-12345
gcloud config set project irh-project-12345

# Enable billing (required for Cloud Run)
# Visit: https://console.cloud.google.com/billing

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Step 2: Deploy with One Command

```bash
# From repository root
./deploy-to-cloudrun.sh YOUR_PROJECT_ID
```

That's it! The script will:
1. Build the Docker image (3-5 minutes)
2. Push to Google Container Registry
3. Deploy to Cloud Run (1-2 minutes)
4. Output your service URL

## Step 3: Test Your Deployment

```bash
# Your service URL will be output (example):
# https://irh-backend-abc123-uc.a.run.app

# Test health endpoint
curl https://YOUR-SERVICE-URL/health

# Test fixed point computation
curl https://YOUR-SERVICE-URL/api/v1/fixed-point

# Test fine-structure constant
curl https://YOUR-SERVICE-URL/api/v1/observables/alpha

# View interactive API docs
open https://YOUR-SERVICE-URL/docs
```

## Expected Output

### Health Check
```json
{
  "status": "healthy",
  "version": "21.0.0",
  "irh_available": true,
  "modules_loaded": ["rg_flow.fixed_points", ...]
}
```

### Fixed Point
```json
{
  "lambda_star": 52.64,
  "gamma_star": 105.28,
  "mu_star": 157.91,
  "C_H": 0.045935703598,
  "theoretical_reference": "IRH v21.1 §1.2-1.3, Eq. 1.14"
}
```

### Fine-Structure Constant
```json
{
  "name": "Fine-Structure Constant α⁻¹",
  "value": 137.035999084,
  "uncertainty": 1e-09,
  "theoretical_reference": "IRH v21.1 §3.2, Eqs. 3.4-3.5"
}
```

## Manual Deployment (Alternative)

If you prefer manual control:

```bash
# Build image with Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/irh-backend

# Deploy to Cloud Run
gcloud run deploy irh-backend \
  --image gcr.io/YOUR_PROJECT_ID/irh-backend \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

## Cost Estimate

For moderate usage:
- **10,000 requests/month**: ~$1-2/month
- **100,000 requests/month**: ~$5-10/month
- **1 million requests/month**: ~$50-100/month

Cloud Run includes:
- 2 million requests/month free
- 360,000 GB-seconds/month free
- 180,000 vCPU-seconds/month free

## Configuration Options

### Set Environment Variables

```bash
gcloud run services update irh-backend \
  --set-env-vars "LOG_LEVEL=DEBUG"
```

### Update Resources

```bash
gcloud run services update irh-backend \
  --memory 4Gi \
  --cpu 4
```

### Enable Authentication

```bash
# Make service private
gcloud run services update irh-backend \
  --no-allow-unauthenticated

# Grant access to specific users
gcloud run services add-iam-policy-binding irh-backend \
  --member="user:someone@example.com" \
  --role="roles/run.invoker"
```

## Troubleshooting

### Build Fails
```bash
# Check Cloud Build logs
gcloud builds log $(gcloud builds list --limit=1 --format='value(id)')
```

### Service Fails to Start
```bash
# View logs
gcloud run services logs read irh-backend --limit 50
```

### Cannot Access Service
```bash
# Check service status
gcloud run services describe irh-backend

# Make service public (if needed)
gcloud run services add-iam-policy-binding irh-backend \
  --member="allUsers" \
  --role="roles/run.invoker"
```

## Next Steps

1. **Custom Domain**: Map your own domain to the service
2. **CI/CD**: Set up automatic deployments from GitHub
3. **Monitoring**: Enable Cloud Monitoring and alerting
4. **Scaling**: Configure min/max instances for your traffic

See full documentation: [`CLOUD_RUN_DEPLOYMENT.md`](./CLOUD_RUN_DEPLOYMENT.md)

## Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [IRH Repository](https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-)

---

**Questions?** Open an issue on GitHub or consult the full deployment guide.
