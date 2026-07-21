#!/bin/bash
set -e

# AWS Deployment Script for Algo Trading Bot
# Deploys to ECS Fargate with ECR for container storage
#
# Prerequisites:
#   - AWS CLI installed and configured (aws configure)
#   - Docker installed
#   - IAM permissions for ECR, ECS, CloudWatch
#
# Usage:
#   ./aws/deploy.sh [region] [cluster-name]
#
# Example:
#   ./aws/deploy.sh us-east-1 algo-trading-prod

REGION="${1:-us-east-1}"
CLUSTER_NAME="${2:-algo-trading}"
SERVICE_NAME="live-trading"
TASK_FAMILY="algotrader"
ECR_REPO="algotrader"
CONTAINER_NAME="algotrader"
CPU="512"      # 0.5 vCPU
MEMORY="1024"  # 1 GB

echo "==========================================="
echo "AWS ECS Deployment"
echo "Region: $REGION"
echo "Cluster: $CLUSTER_NAME"
echo "==========================================="

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_URL}/${ECR_REPO}:latest"

echo "✓ AWS Account: $ACCOUNT_ID"

# Step 1: Create ECR repository (if not exists)
echo ""
echo "→ Creating ECR repository..."
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true

echo "✓ ECR repository ready: $ECR_REPO"

# Step 2: Login to ECR
echo ""
echo "→ Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URL"
echo "✓ Logged in to ECR"

# Step 3: Build and push Docker image
echo ""
echo "→ Building Docker image..."
docker build -t "$ECR_REPO:latest" .
docker tag "$ECR_REPO:latest" "$IMAGE_URI"

echo "→ Pushing to ECR..."
docker push "$IMAGE_URI"
echo "✓ Image pushed: $IMAGE_URI"

# Step 4: Create ECS cluster (if not exists)
echo ""
echo "→ Creating ECS cluster..."
aws ecs describe-clusters --clusters "$CLUSTER_NAME" --region "$REGION" --query "clusters[0].status" --output text 2>/dev/null | grep -q "ACTIVE" || \
  aws ecs create-cluster --cluster-name "$CLUSTER_NAME" --region "$REGION"
echo "✓ Cluster ready: $CLUSTER_NAME"

# Step 5: Create CloudWatch log group (if not exists)
LOG_GROUP="/ecs/${TASK_FAMILY}"
echo ""
echo "→ Creating CloudWatch log group..."
aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --region "$REGION" 2>/dev/null | grep -q "$LOG_GROUP" || \
  aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION"
aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 7 --region "$REGION" 2>/dev/null || true
echo "✓ Log group ready: $LOG_GROUP"

# Step 6: Create task execution role (if not exists)
ROLE_NAME="ecsTaskExecutionRole"
echo ""
echo "→ Checking IAM role..."
aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null || {
  echo "Creating $ROLE_NAME..."
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document file://aws/task-execution-role-trust-policy.json
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
  sleep 5  # Wait for IAM propagation
}
EXECUTION_ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query "Role.Arn" --output text)
echo "✓ Execution role: $EXECUTION_ROLE_ARN"

# Step 7: Get secrets from SSM Parameter Store
echo ""
echo "→ Fetching secrets from SSM Parameter Store..."
echo "   (Create these with: aws ssm put-parameter --name /algotrader/ALPACA_API_KEY --value YOUR_KEY --type SecureString)"

# Check if secrets exist
ALPACA_KEY_PARAM="/algotrader/ALPACA_API_KEY"
ALPACA_SECRET_PARAM="/algotrader/ALPACA_API_SECRET"

if ! aws ssm get-parameter --name "$ALPACA_KEY_PARAM" --region "$REGION" &>/dev/null; then
  echo ""
  echo "⚠️  ERROR: Alpaca API credentials not found in SSM Parameter Store"
  echo ""
  echo "Please create them with:"
  echo ""
  echo "  aws ssm put-parameter --name '$ALPACA_KEY_PARAM' --value 'YOUR_API_KEY' --type SecureString --region '$REGION'"
  echo "  aws ssm put-parameter --name '$ALPACA_SECRET_PARAM' --value 'YOUR_API_SECRET' --type SecureString --region '$REGION'"
  echo ""
  exit 1
fi
echo "✓ Secrets configured in Parameter Store"

# Step 8: Register task definition
echo ""
echo "→ Registering ECS task definition..."
cat > /tmp/task-def.json <<EOF
{
  "family": "$TASK_FAMILY",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "$CPU",
  "memory": "$MEMORY",
  "executionRoleArn": "$EXECUTION_ROLE_ARN",
  "containerDefinitions": [
    {
      "name": "$CONTAINER_NAME",
      "image": "$IMAGE_URI",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ALPACA_PAPER", "value": "true"},
        {"name": "LIVE_PARAMS_PATH", "value": "config/trial_432_params.json"}
      ],
      "secrets": [
        {
          "name": "ALPACA_API_KEY",
          "valueFrom": "arn:aws:ssm:${REGION}:${ACCOUNT_ID}:parameter${ALPACA_KEY_PARAM}"
        },
        {
          "name": "ALPACA_API_SECRET",
          "valueFrom": "arn:aws:ssm:${REGION}:${ACCOUNT_ID}:parameter${ALPACA_SECRET_PARAM}"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "$LOG_GROUP",
          "awslogs-region": "$REGION",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c 'import urllib.request; urllib.request.urlopen(\"http://localhost:8000/api/account\")' || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
EOF

TASK_DEF_ARN=$(aws ecs register-task-definition \
  --cli-input-json file:///tmp/task-def.json \
  --region "$REGION" \
  --query "taskDefinition.taskDefinitionArn" \
  --output text)
echo "✓ Task definition registered: $TASK_DEF_ARN"

# Step 9: Get default VPC and subnets
echo ""
echo "→ Finding VPC and subnets..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
  --query "Vpcs[0].VpcId" --output text --region "$REGION")

if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
  echo "⚠️  No default VPC found. Creating one..."
  aws ec2 create-default-vpc --region "$REGION"
  VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
    --query "Vpcs[0].VpcId" --output text --region "$REGION")
fi

SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "Subnets[*].SubnetId" --output text --region "$REGION" | tr '\t' ',')

echo "✓ VPC: $VPC_ID"
echo "✓ Subnets: $SUBNETS"

# Step 10: Create security group (if not exists)
SG_NAME="algotrader-sg"
SG_DESC="Security group for algo trading bot"
echo ""
echo "→ Creating security group..."
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text --region "$REGION" 2>/dev/null)

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
  SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "$SG_DESC" \
    --vpc-id "$VPC_ID" --region "$REGION" --query "GroupId" --output text)

  # Allow inbound HTTP (8000) - only if you want external dashboard access
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region "$REGION" 2>/dev/null || true

  # Allow outbound (default allows all)
fi
echo "✓ Security group: $SG_ID"

# Step 11: Create or update ECS service
echo ""
echo "→ Creating/updating ECS service..."
SERVICE_EXISTS=$(aws ecs describe-services --cluster "$CLUSTER_NAME" --services "$SERVICE_NAME" \
  --region "$REGION" --query "services[0].status" --output text 2>/dev/null)

if [ "$SERVICE_EXISTS" = "ACTIVE" ]; then
  echo "Updating existing service..."
  aws ecs update-service \
    --cluster "$CLUSTER_NAME" \
    --service "$SERVICE_NAME" \
    --task-definition "$TASK_DEF_ARN" \
    --region "$REGION" \
    --force-new-deployment > /dev/null
else
  echo "Creating new service..."
  aws ecs create-service \
    --cluster "$CLUSTER_NAME" \
    --service-name "$SERVICE_NAME" \
    --task-definition "$TASK_DEF_ARN" \
    --desired-count 1 \
    --launch-type FARGATE \
    --platform-version LATEST \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --region "$REGION" > /dev/null
fi

echo "✓ Service deployed: $SERVICE_NAME"

# Step 12: Get public IP
echo ""
echo "→ Waiting for task to start (this may take 60-90 seconds)..."
sleep 15

TASK_ARN=$(aws ecs list-tasks --cluster "$CLUSTER_NAME" --service-name "$SERVICE_NAME" \
  --region "$REGION" --query "taskArns[0]" --output text)

if [ "$TASK_ARN" != "None" ] && [ -n "$TASK_ARN" ]; then
  ENI_ID=$(aws ecs describe-tasks --cluster "$CLUSTER_NAME" --tasks "$TASK_ARN" \
    --region "$REGION" --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
    --output text)

  if [ -n "$ENI_ID" ] && [ "$ENI_ID" != "None" ]; then
    PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI_ID" \
      --region "$REGION" --query "NetworkInterfaces[0].Association.PublicIp" --output text)

    echo ""
    echo "==========================================="
    echo "✓ DEPLOYMENT COMPLETE"
    echo "==========================================="
    echo ""
    echo "Dashboard URL: http://${PUBLIC_IP}:8000"
    echo ""
    echo "Logs:"
    echo "  aws logs tail '$LOG_GROUP' --follow --region $REGION"
    echo ""
    echo "Stop service:"
    echo "  aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count 0 --region $REGION"
    echo ""
  fi
fi

echo "Service status:"
echo "  aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION"
echo ""
