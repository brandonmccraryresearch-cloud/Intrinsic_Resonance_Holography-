# IRH v21.1 Deployment Guide

This directory contains Docker and Kubernetes configurations for deploying the Intrinsic Resonance Holography (IRH) web application.

## Architecture

```
                    ┌─────────────────┐
                    │     Ingress     │
                    │  (HTTPS/TLS)    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
        ┌─────▼─────┐                 ┌─────▼─────┐
        │  Frontend │                 │  Backend  │
        │  (nginx)  │                 │ (FastAPI) │
        │  :3000    │────────────────▶│  :8000    │
        └───────────┘      /api       └───────────┘
```

## Docker Deployment

### Quick Start

```bash
cd deploy/docker
docker-compose up -d
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `irh-backend` | 8000 | FastAPI REST API |
| `irh-frontend` | 3000 | React frontend (nginx) |
| `irh-redis` | 6379 | Redis cache (optional) |

### Build Images Manually

```bash
# Build backend
docker build -f deploy/docker/Dockerfile.backend -t irh-backend:latest .

# Build frontend
docker build -f deploy/docker/Dockerfile.frontend -t irh-frontend:latest .
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | `/app` | Python module path |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API port |

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured
- nginx-ingress controller
- cert-manager (for TLS)

### Deploy

```bash
# Create namespace
kubectl apply -f deploy/kubernetes/namespace.yaml

# Deploy configmap
kubectl apply -f deploy/kubernetes/configmap.yaml

# Deploy backend
kubectl apply -f deploy/kubernetes/backend-deployment.yaml

# Deploy frontend
kubectl apply -f deploy/kubernetes/frontend-deployment.yaml

# Configure ingress
kubectl apply -f deploy/kubernetes/ingress.yaml

# Enable autoscaling
kubectl apply -f deploy/kubernetes/hpa.yaml
```

### Verify Deployment

```bash
# Check pods
kubectl get pods -n irh

# Check services
kubectl get svc -n irh

# Check ingress
kubectl get ingress -n irh

# View logs
kubectl logs -f deployment/irh-backend -n irh
```

### Configuration

Update `deploy/kubernetes/configmap.yaml` with your settings:

```yaml
data:
  LOG_LEVEL: "INFO"
  IRH_VERSION: "21.1.0"
  # ...
```

Update `deploy/kubernetes/ingress.yaml` with your domain:

```yaml
spec:
  tls:
  - hosts:
    - your-domain.com
```

### Scaling

Horizontal Pod Autoscaler is configured in `hpa.yaml`:

- **Backend**: 2-10 replicas based on CPU/memory
- **Frontend**: 2-5 replicas based on CPU

Manual scaling:

```bash
kubectl scale deployment irh-backend --replicas=5 -n irh
```

## Production Checklist

- [ ] Update `irh.example.com` in ingress to your domain
- [ ] Configure TLS certificates (cert-manager)
- [ ] Set production secrets in `configmap.yaml`
- [ ] Configure resource limits based on load testing
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation (ELK/Loki)
- [ ] Set up backup for Redis data

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/fixed-point` | GET | Cosmic Fixed Point |
| `/api/v1/rg-flow` | POST | RG flow integration |
| `/api/v1/observables/alpha` | GET | Fine-structure constant |
| `/api/v1/observables/C_H` | GET | Universal exponent |
| `/api/v1/observables/dark-energy` | GET | Dark energy EoS |
| `/api/v1/observables/liv` | GET | LIV parameter |
| `/api/v1/standard-model/gauge-group` | GET | Gauge group derivation |
| `/api/v1/standard-model/neutrinos` | GET | Neutrino predictions |
| `/api/v1/falsification/summary` | GET | Falsification status |
| `/docs` | GET | Swagger documentation |

## Resource Requirements

### Minimum (Development)

- Backend: 512MB RAM, 0.5 CPU
- Frontend: 64MB RAM, 0.1 CPU

### Recommended (Production)

- Backend: 2GB RAM, 2 CPU (per replica)
- Frontend: 256MB RAM, 0.2 CPU (per replica)

## Monitoring

### Health Endpoints

```bash
# Backend health
curl http://localhost:8000/health

# Frontend (nginx)
curl http://localhost:3000/
```

### Metrics (planned)

- Prometheus metrics at `/metrics`
- Grafana dashboards

## Troubleshooting

### Pod not starting

```bash
kubectl describe pod <pod-name> -n irh
kubectl logs <pod-name> -n irh
```

### Cannot connect to backend

```bash
kubectl exec -it <frontend-pod> -n irh -- curl http://irh-backend:8000/health
```

### Reset deployment

```bash
kubectl delete namespace irh
kubectl apply -f deploy/kubernetes/
```
