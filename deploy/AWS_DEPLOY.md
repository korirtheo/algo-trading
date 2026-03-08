# AWS Deployment Guide

## 1. Launch EC2 Instance

1. Go to **AWS Console > EC2 > Launch Instance**
2. Settings:
   - **Name**: `algotrader`
   - **AMI**: Ubuntu 22.04 LTS (free tier eligible)
   - **Instance type**: `t2.micro` (free tier) or `t3.micro`
   - **Key pair**: Create new or use existing (download .pem file)
   - **Security Group**: Create new with these rules:
     - SSH (port 22) — your IP only
     - HTTP (port 80) — anywhere (0.0.0.0/0)
   - **Storage**: 20 GB gp3 (free tier allows up to 30 GB)
3. Click **Launch Instance**
4. Note the **Public IP** from the instance details

## 2. Connect & Setup

```bash
# SSH into instance (replace with your .pem path and IP)
ssh -i "your-key.pem" ubuntu@<EC2-PUBLIC-IP>

# Install Docker
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker

# Install Git
sudo apt-get install -y git
```

## 3. Deploy

```bash
# Clone your repo
git clone <your-repo-url> algo-trading
cd algo-trading

# Create .env with your Alpaca keys
cat > .env << 'EOF'
ALPACA_API_KEY=PK34OGXUBAOCLG7E6KYTI6QMZ3
ALPACA_API_SECRET=DBn7mXAKdTBAR9XZnkhnu1CykZDYNZEVzkBojKDtoYbJ
ALPACA_PAPER=true
EOF

# Build and run
docker compose up -d --build

# Check logs
docker compose logs -f
```

## 4. Access Dashboard

Open browser: `http://<EC2-PUBLIC-IP>`

## Useful Commands

```bash
docker compose logs -f          # live logs
docker compose restart           # restart
docker compose down              # stop
docker compose up -d --build     # rebuild after code changes
docker compose exec algotrader python -c "print('ok')"  # shell into container
```

## Costs

- **t2.micro**: Free for 12 months (750 hrs/month)
- **After free tier**: ~$8.50/month for t3.micro
- **Storage**: 20 GB free (first 12 months)
