# MoneyPrinterTurbo Deployment Architecture

## 1. Container Orchestration Strategy

### 1.1 Docker Containerization

Each microservice is containerized with optimized Docker images:

```dockerfile
# Base Dockerfile Template for FastAPI Services
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Service-specific configurations
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 1.2 Kubernetes Deployment

#### Production Kubernetes Configuration

```yaml
# api-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: moneyprinter
  labels:
    app: api-gateway
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
        version: v1.0.0
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: api-gateway
        image: moneyprinter/api-gateway:1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: jwt-secret
              key: secret
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
  namespace: moneyprinter
spec:
  selector:
    app: api-gateway
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP

---
# video-processing-deployment.yaml (GPU-enabled)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: video-processing
  namespace: moneyprinter
spec:
  replicas: 2
  selector:
    matchLabels:
      app: video-processing
  template:
    metadata:
      labels:
        app: video-processing
    spec:
      nodeSelector:
        gpu: "true"
      containers:
      - name: video-processing
        image: moneyprinter/video-processing:1.0.0
        ports:
        - containerPort: 8004
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: FFMPEG_GPU_ACCELERATION
          value: "true"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: video-storage
          mountPath: /tmp/video_processing
        - name: gpu-device
          mountPath: /dev/nvidia0
      volumes:
      - name: video-storage
        persistentVolumeClaim:
          claimName: video-processing-pvc
      - name: gpu-device
        hostPath:
          path: /dev/nvidia0

---
# PostgreSQL StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: moneyprinter
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: moneyprinter
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
      storageClassName: fast-ssd

---
# Redis Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: moneyprinter
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "--appendonly", "yes"]
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc
```

### 1.3 Helm Charts

```yaml
# Chart.yaml
apiVersion: v2
name: moneyprinter
description: MoneyPrinterTurbo Microservices Platform
type: application
version: 1.0.0
appVersion: "1.0.0"

dependencies:
  - name: postgresql
    version: "12.1.2"
    repository: "https://charts.bitnami.com/bitnami"
  - name: redis
    version: "17.3.7"
    repository: "https://charts.bitnami.com/bitnami"
  - name: nginx-ingress
    version: "4.4.0"
    repository: "https://kubernetes.github.io/ingress-nginx"

# values.yaml
global:
  imageRegistry: "registry.moneyprinter.com"
  imagePullPolicy: IfNotPresent
  storageClass: "fast-ssd"

services:
  apiGateway:
    enabled: true
    replicaCount: 3
    image:
      repository: api-gateway
      tag: "1.0.0"
    resources:
      requests:
        memory: "256Mi"
        cpu: "250m"
      limits:
        memory: "512Mi"
        cpu: "500m"

  userService:
    enabled: true
    replicaCount: 2
    image:
      repository: user-service
      tag: "1.0.0"
    resources:
      requests:
        memory: "256Mi"
        cpu: "200m"
      limits:
        memory: "512Mi"
        cpu: "400m"

  contentService:
    enabled: true
    replicaCount: 2
    image:
      repository: content-service
      tag: "1.0.0"
    resources:
      requests:
        memory: "512Mi"
        cpu: "500m"
      limits:
        memory: "1Gi"
        cpu: "1000m"

  ttsService:
    enabled: true
    replicaCount: 3
    image:
      repository: tts-service
      tag: "1.0.0"
    resources:
      requests:
        memory: "512Mi"
        cpu: "500m"
      limits:
        memory: "1Gi"
        cpu: "1000m"

  videoProcessingService:
    enabled: true
    replicaCount: 2
    image:
      repository: video-processing
      tag: "1.0.0"
    nodeSelector:
      gpu: "true"
    resources:
      requests:
        memory: "4Gi"
        cpu: "2000m"
        nvidia.com/gpu: 1
      limits:
        memory: "8Gi"
        cpu: "4000m"
        nvidia.com/gpu: 1

  materialService:
    enabled: true
    replicaCount: 2
    image:
      repository: material-service
      tag: "1.0.0"
    resources:
      requests:
        memory: "512Mi"
        cpu: "300m"
      limits:
        memory: "1Gi"
        cpu: "600m"

  orchestrationService:
    enabled: true
    replicaCount: 2
    image:
      repository: orchestration-service
      tag: "1.0.0"
    resources:
      requests:
        memory: "512Mi"
        cpu: "300m"
      limits:
        memory: "1Gi"
        cpu: "600m"

  notificationService:
    enabled: true
    replicaCount: 2
    image:
      repository: notification-service
      tag: "1.0.0"
    resources:
      requests:
        memory: "256Mi"
        cpu: "200m"
      limits:
        memory: "512Mi"
        cpu: "400m"

ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/rate-limit-requests-per-second: "10"
    nginx.ingress.kubernetes.io/rate-limit-burst-multiplier: "5"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: api.moneyprinter.com
      paths:
        - path: /
          pathType: Prefix
          service:
            name: api-gateway-service
            port: 80
  tls:
    - secretName: api-tls
      hosts:
        - api.moneyprinter.com

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
  jaeger:
    enabled: true

storage:
  objectStore:
    provider: "minio"
    endpoint: "minio.moneyprinter.com"
    buckets:
      - name: "user-assets"
        policy: "private"
      - name: "generated-videos"
        policy: "private"
      - name: "stock-materials"
        policy: "public-read"
```

## 2. Infrastructure as Code (Terraform)

### 2.1 AWS Infrastructure

```hcl
# main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "moneyprinter-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = true
  
  tags = {
    Environment = var.environment
    Project     = "MoneyPrinter"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "moneyprinter-${var.environment}"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # GPU Node Group for Video Processing
  eks_managed_node_groups = {
    gpu_nodes = {
      desired_size = 2
      max_size     = 4
      min_size     = 1
      
      instance_types = ["p3.2xlarge"]
      
      k8s_labels = {
        gpu = "true"
        workload = "video-processing"
      }
      
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
    
    general_nodes = {
      desired_size = 3
      max_size     = 10
      min_size     = 3
      
      instance_types = ["t3.large", "t3.xlarge"]
      
      k8s_labels = {
        workload = "general"
      }
    }
  }
  
  tags = {
    Environment = var.environment
    Project     = "MoneyPrinter"
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier = "moneyprinter-${var.environment}"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.large"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "moneyprinter"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.postgres.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = var.environment != "prod"
  
  tags = {
    Environment = var.environment
    Project     = "MoneyPrinter"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "redis" {
  name       = "moneyprinter-redis-${var.environment}"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "moneyprinter-${var.environment}"
  description                = "Redis cluster for MoneyPrinter"
  
  node_type                  = "cache.t3.medium"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  
  subnet_group_name = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Environment = var.environment
    Project     = "MoneyPrinter"
  }
}

# S3 Buckets for Asset Storage
resource "aws_s3_bucket" "assets" {
  bucket = "moneyprinter-assets-${var.environment}-${random_id.bucket_suffix.hex}"
  
  tags = {
    Environment = var.environment
    Project     = "MoneyPrinter"
  }
}

resource "aws_s3_bucket_versioning" "assets" {
  bucket = aws_s3_bucket.assets.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "assets" {
  bucket = aws_s3_bucket.assets.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# CloudFront CDN
resource "aws_cloudfront_distribution" "assets_cdn" {
  origin {
    domain_name = aws_s3_bucket.assets.bucket_regional_domain_name
    origin_id   = "S3-${aws_s3_bucket.assets.id}"
    
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.assets.cloudfront_access_identity_path
    }
  }
  
  enabled = true
  
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3-${aws_s3_bucket.assets.id}"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = true
  }
  
  tags = {
    Environment = var.environment
    Project     = "MoneyPrinter"
  }
}

# Security Groups
resource "aws_security_group" "rds" {
  name_prefix = "moneyprinter-rds-${var.environment}"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Environment = var.environment
    Project     = "MoneyPrinter"
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "moneyprinter-redis-${var.environment}"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  tags = {
    Environment = var.environment
    Project     = "MoneyPrinter"
  }
}

# Random ID for unique resource naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Outputs
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "rds_endpoint" {
  value = aws_db_instance.postgres.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.redis.primary_endpoint
}

output "cdn_domain" {
  value = aws_cloudfront_distribution.assets_cdn.domain_name
}
```

### 2.2 Variables and Environment Configuration

```hcl
# variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "db_username" {
  description = "Database username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

# terraform.tfvars.example
aws_region  = "us-west-2"
environment = "dev"
db_username = "moneyprinter"
db_password = "your-secure-password"
```

## 3. CI/CD Pipeline

### 3.1 GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy MoneyPrinter Microservices

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_PREFIX: moneyprinter

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    strategy:
      matrix:
        service:
          - api-gateway
          - user-service
          - content-service
          - tts-service
          - video-processing
          - material-service
          - orchestration-service
          - notification-service
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/${{ matrix.service }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: ./services/${{ matrix.service }}
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --name moneyprinter-staging --region us-west-2
    
    - name: Deploy with Helm
      run: |
        helm upgrade --install moneyprinter ./helm/moneyprinter \
          --namespace moneyprinter-staging \
          --create-namespace \
          --values ./helm/values-staging.yaml \
          --set global.imageTag=${{ github.sha }}

  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --name moneyprinter-prod --region us-west-2
    
    - name: Deploy with Helm (Blue-Green)
      run: |
        # Deploy to green environment
        helm upgrade --install moneyprinter-green ./helm/moneyprinter \
          --namespace moneyprinter-prod \
          --values ./helm/values-prod.yaml \
          --set global.imageTag=${{ github.sha }} \
          --set global.environment=green
        
        # Health check
        kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=moneyprinter-green \
          --namespace moneyprinter-prod --timeout=300s
        
        # Switch traffic (blue-green deployment)
        kubectl patch service api-gateway-service \
          --namespace moneyprinter-prod \
          --patch '{"spec":{"selector":{"app.kubernetes.io/instance":"moneyprinter-green"}}}'
        
        # Clean up old deployment after successful switch
        helm uninstall moneyprinter-blue --namespace moneyprinter-prod || true
```

## 4. Monitoring and Observability

### 4.1 Prometheus Configuration

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'moneyprinter-services'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - moneyprinter
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

  - job_name: 'gpu-metrics'
    static_configs:
      - targets: ['gpu-exporter:9445']
    scrape_interval: 10s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 4.2 Grafana Dashboards

```json
{
  "dashboard": {
    "title": "MoneyPrinter Microservices Overview",
    "panels": [
      {
        "title": "Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"moneyprinter-services\"}",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{service}} - {{method}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Video Processing Queue",
        "type": "stat",
        "targets": [
          {
            "expr": "celery_queue_length{queue=\"video_processing\"}",
            "legendFormat": "Queue Length"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      }
    ]
  }
}
```

### 4.3 Alert Rules

```yaml
# alerts.yml
groups:
  - name: moneyprinter.rules
    rules:
      - alert: ServiceDown
        expr: up{job="moneyprinter-services"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.instance }} is down"
          description: "{{ $labels.instance }} has been down for more than 1 minute"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time for {{ $labels.service }}"
          description: "95th percentile response time is {{ $value }}s"

      - alert: VideoProcessingQueueBacklog
        expr: celery_queue_length{queue="video_processing"} > 50
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Video processing queue backlog"
          description: "Queue length is {{ $value }}, consider scaling up"

      - alert: GPUUtilizationHigh
        expr: nvidia_gpu_utilization_gpu > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU utilization high on {{ $labels.instance }}"
          description: "GPU utilization is {{ $value }}%"

      - alert: DatabaseConnections
        expr: pg_stat_database_numbackends / pg_settings_max_connections * 100 > 80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High database connection usage"
          description: "Database connection usage is {{ $value }}%"
```

## 5. Security and Compliance

### 5.1 Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: moneyprinter-network-policy
  namespace: moneyprinter
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8001
    - protocol: TCP
      port: 8002
    - protocol: TCP
      port: 8003
    - protocol: TCP
      port: 8004
    - protocol: TCP
      port: 8005
    - protocol: TCP
      port: 8006
    - protocol: TCP
      port: 8007
  
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  
  - to: []
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 443   # HTTPS
```

### 5.2 Pod Security Standards

```yaml
# pod-security-policy.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: moneyprinter
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted

---
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: moneyprinter-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  supplementalGroups:
    rule: 'MustRunAs'
    ranges:
      - min: 1
        max: 65535
  fsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1
        max: 65535
  readOnlyRootFilesystem: true
```

This deployment architecture provides a production-ready, scalable, and secure foundation for the MoneyPrinterTurbo microservices platform with comprehensive monitoring, automated deployments, and infrastructure as code.