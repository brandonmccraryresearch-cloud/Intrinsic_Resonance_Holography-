# Google Cloud Run - Quick Reference Card

## 🚀 Deploy in 3 Steps

```bash
# 1. Enable APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# 2. Deploy
./deploy-to-cloudrun.sh YOUR_PROJECT_ID

# 3. Test
curl https://YOUR-SERVICE-URL/health
```

## 📋 Common Commands

### Deployment
```bash
# Quick deploy
./deploy-to-cloudrun.sh PROJECT_ID

# Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Manual deploy
gcloud run deploy irh-backend \
  --image gcr.io/PROJECT/irh-backend \
  --region us-central1 \
  --allow-unauthenticated
```

### Management
```bash
# Get service URL
gcloud run services describe irh-backend \
  --region us-central1 \
  --format 'value(status.url)'

# View logs
gcloud run services logs tail irh-backend

# Update service
gcloud run services update irh-backend \
  --memory 4Gi --cpu 4

# Delete service
gcloud run services delete irh-backend
```

### Configuration
```bash
# Set environment variables
gcloud run services update irh-backend \
  --set-env-vars "LOG_LEVEL=DEBUG,VAR=value"

# Update resources
gcloud run services update irh-backend \
  --memory 2Gi --cpu 2 --concurrency 80

# Update scaling
gcloud run services update irh-backend \
  --min-instances 1 --max-instances 10

# Update timeout
gcloud run services update irh-backend \
  --timeout 300s
```

### Security
```bash
# Make private
gcloud run services update irh-backend \
  --no-allow-unauthenticated

# Grant access
gcloud run services add-iam-policy-binding irh-backend \
  --member="user:email@example.com" \
  --role="roles/run.invoker"

# Make public
gcloud run services add-iam-policy-binding irh-backend \
  --member="allUsers" \
  --role="roles/run.invoker"
```

## 📊 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check |
| `/docs` | API documentation |
| `/api/v1/fixed-point` | Cosmic Fixed Point |
| `/api/v1/observables/alpha` | Fine-structure constant |
| `/api/v1/observables/dark-energy` | Dark energy EoS |
| `/api/v1/standard-model/gauge-group` | Gauge group |

## 💰 Cost Estimates

| Usage | Cost/Month |
|-------|------------|
| 10K requests | $1-2 |
| 100K requests | $5-10 |
| 1M requests | $50-100 |

**Free Tier:**
- 2M requests/month
- 360K GB-seconds
- 180K vCPU-seconds

## ⚙️ Default Configuration

```yaml
Region: us-central1
Memory: 2Gi
CPU: 2
Concurrency: 80
Min Instances: 0
Max Instances: 10
Timeout: 60s
Port: Dynamic (via $PORT env var)
```

## 🐛 Troubleshooting

### Build Fails
```bash
gcloud builds log $(gcloud builds list --limit=1 --format='value(id)')
```

### Service Unhealthy
```bash
gcloud run services logs read irh-backend --limit 50
```

### Cannot Access
```bash
# Check status
gcloud run services describe irh-backend

# Make public
gcloud run services add-iam-policy-binding irh-backend \
  --member="allUsers" --role="roles/run.invoker"
```

### Timeout Issues
```bash
# Increase timeout
gcloud run services update irh-backend --timeout 300s
```

## 📖 Documentation

- **Quick Start**: `QUICKSTART_CLOUD_RUN.md`
- **Full Guide**: `CLOUD_RUN_DEPLOYMENT.md`
- **Summary**: `DEPLOYMENT_SUMMARY.md`
- **Cloud Run Docs**: https://cloud.google.com/run/docs

## ✅ Pre-Flight Checklist

- [ ] gcloud CLI installed
- [ ] Authenticated (`gcloud auth login`)
- [ ] Project selected (`gcloud config set project PROJECT_ID`)
- [ ] APIs enabled (cloudbuild, run, containerregistry)
- [ ] Billing enabled on project

## 🎯 Quick Test

```bash
SERVICE_URL=$(gcloud run services describe irh-backend \
  --region us-central1 --format 'value(status.url)')

# Health
curl $SERVICE_URL/health

# Fixed Point
curl $SERVICE_URL/api/v1/fixed-point | python3 -m json.tool

# Alpha
curl $SERVICE_URL/api/v1/observables/alpha | python3 -m json.tool

# Docs
open $SERVICE_URL/docs
```

## 🔧 Local Testing

```bash
# Build
docker build -f Dockerfile.cloudrun -t irh-backend:test .

# Run
docker run -p 8080:8080 -e PORT=8080 irh-backend:test

# Test
curl http://localhost:8080/health
```

## 📝 Environment Variables

```bash
PYTHONPATH=/app         # Set automatically
PORT=${PORT}            # Cloud Run injects
LOG_LEVEL=INFO          # Optional, defaults to INFO
```

## 🌐 Custom Domain

```bash
# Verify domain
gcloud domains verify example.com

# Map domain
gcloud run domain-mappings create \
  --service irh-backend \
  --domain api.example.com \
  --region us-central1
```

## 🔄 CI/CD with GitHub

```bash
# Create trigger
gcloud builds triggers create github \
  --repo-name=Intrinsic_Resonance_Holography- \
  --repo-owner=brandonmccraryresearch-cloud \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml
```

## 📈 Monitoring

```bash
# View metrics in console
gcloud run services describe irh-backend \
  --region us-central1

# Real-time logs
gcloud run services logs tail irh-backend \
  --region us-central1

# Metrics dashboard
open "https://console.cloud.google.com/run/detail/us-central1/irh-backend/metrics"
```

## 🎉 Success Indicators

✅ Health check returns 200 OK  
✅ `/api/v1/fixed-point` returns λ̃* ≈ 52.64  
✅ `/api/v1/observables/alpha` returns 137.035999084  
✅ `/docs` shows Swagger UI  
✅ Service URL accessible globally

---

**Need Help?** See [`CLOUD_RUN_DEPLOYMENT.md`](./CLOUD_RUN_DEPLOYMENT.md) for detailed guide.
