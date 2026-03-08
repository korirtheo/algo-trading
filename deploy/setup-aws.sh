#!/bin/bash
# AWS EC2 setup script — run this after SSH into a fresh Ubuntu 22.04 instance
# Usage: curl -sSL <raw-gist-url> | bash
# Or: scp this file to the instance and run: bash setup-aws.sh

set -e

echo "=== AlgoTrader AWS Setup ==="

# 1. Install Docker
echo "Installing Docker..."
sudo apt-get update -qq
sudo apt-get install -y -qq ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
echo "Docker installed."

# 2. Install Git
sudo apt-get install -y -qq git

# 3. Clone repo
echo ""
echo "Clone your repo:"
echo "  git clone <your-repo-url> algo-trading"
echo "  cd algo-trading"
echo ""

# 4. Create .env file
echo "Create .env file with your Alpaca keys:"
echo '  cat > .env << EOF'
echo '  ALPACA_API_KEY=your_key_here'
echo '  ALPACA_API_SECRET=your_secret_here'
echo '  ALPACA_PAPER=true'
echo '  EOF'
echo ""

# 5. Build and run
echo "Then build and run:"
echo "  sudo docker compose up -d --build"
echo ""
echo "Dashboard will be at http://<your-ec2-public-ip>"
echo ""
echo "=== Important: Open port 80 in your EC2 Security Group ==="
echo "AWS Console > EC2 > Security Groups > Edit Inbound Rules"
echo "  Add rule: Type=HTTP, Port=80, Source=0.0.0.0/0"
echo ""
echo "=== Useful commands ==="
echo "  docker compose logs -f          # view live logs"
echo "  docker compose restart           # restart"
echo "  docker compose down              # stop"
echo "  docker compose up -d --build     # rebuild & restart"
