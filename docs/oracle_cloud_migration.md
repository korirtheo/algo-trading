# Oracle Cloud Migration Runbook

Migrating the algo-trading bot from AWS (suspended) to Oracle Cloud
Infrastructure (OCI) Always Free tier.

**Target architecture**: ARM Ampere A1 VM (4 OCPU, 24 GB RAM, 200 GB SSD,
Ubuntu 22.04, Ashburn US-East region — closest to Alpaca NJ datacenter).

---

## Phase 1: Oracle account + VM provisioning (~30 min, your turn)

### 1.1 Sign up for Oracle Cloud
1. Go to <https://www.oracle.com/cloud/free/>
2. Click "Start for free"
3. **Important**: when choosing the home region, pick **US East (Ashburn)** — lowest latency to Alpaca's NJ data center.
4. Credit card required for identity verification (won't be charged if you stay in Always Free).
5. Wait for account approval (~10-30 min).

### 1.2 Create the VM (Compute → Instances → Create Instance)

| Setting | Value |
|---|---|
| Name | `algotrader` |
| **Image** | Ubuntu 22.04 (Canonical) — must be Ubuntu, not Oracle Linux |
| **Shape** | `VM.Standard.A1.Flex` — **the ARM Ampere shape** |
| **OCPU** | 4 |
| **Memory** | 24 GB |
| **Boot volume** | Default 50 GB is fine (we can grow later if needed) |
| **VCN/Subnet** | Default (or create new) |
| **Public IP** | ✅ Assign a public IPv4 |
| **SSH keys** | Upload your local public key, OR let Oracle generate a keypair and download the private key |

⚠️ **If you get "Out of host capacity"** — the ARM shape is constrained. Try a different availability domain in the same region, or retry in a few hours. Don't pay for a "fixed" instance — it's a temporary capacity issue.

### 1.3 Open the firewall port

In OCI Console:
1. Networking → Virtual Cloud Networks → your VCN → Security Lists → default
2. Add Ingress Rule:
   - **Source CIDR**: `0.0.0.0/0`
   - **Protocol**: TCP
   - **Destination port**: `80`
3. (Optional) Same for port 8000 if you want direct dashboard access without nginx.

### 1.4 Get the public IP

Note down the **Public IP address** of the new instance — you'll need it for SSH.

---

## Phase 2: SSH access (5 min)

If Oracle generated the key for you, save the downloaded `ssh-key-YYYY-MM-DD.key`
to your local machine. Otherwise, your existing `~/.ssh/id_rsa` (or whatever
public key you uploaded) will work.

Test SSH:
```bash
chmod 600 ssh-key-YYYY-MM-DD.key   # if downloaded from Oracle
ssh -i ssh-key-YYYY-MM-DD.key ubuntu@<NEW_PUBLIC_IP>
```

⚠️ Oracle's default Ubuntu image **blocks port 80 in `iptables`** even after
you open it in the VCN security list. Run this once on the new VM:
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo netfilter-persistent save
```

---

## Phase 3: Install Docker + clone the repo (10 min)

SSH in as `ubuntu` and run:

```bash
# System update
sudo apt update && sudo apt upgrade -y

# Docker install (official)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
# Log out + log back in for the group to take effect
exit
```

SSH back in, then:

```bash
# Confirm Docker works without sudo
docker ps

# Clone the repo
cd ~
git clone https://github.com/korirtheo/algo-trading.git
cd algo-trading

# Verify expected files
ls -la docker-compose.yml Dockerfile live/
```

---

## Phase 4: Configure `.env` with current Alpaca keys (5 min)

Create `~/algo-trading/.env`:

```bash
cat > .env <<'EOF'
ALPACA_API_KEY=PKF56A65KVUCGU4DBDPIYRKHNC
ALPACA_API_SECRET=7pmp8Nk3dkZqWweqkAsB9Rkg6QDbBn4YrKfxv6V5Q14p
ALPACA_PAPER=true
LIVE_PARAMS_PATH=config/trial_6_extracted.json
EOF
chmod 600 .env
```

Verify:
```bash
cat .env
```

---

## Phase 5: Build + start the container (5-10 min)

The Docker image is multi-stage (Node for frontend build, Python for runtime).
Both base images are multi-arch on Docker Hub, so the build "just works" on ARM.

```bash
cd ~/algo-trading
sudo docker compose up -d --build
```

First build will take ~5-8 minutes (longer than AWS x86 because Node has to
build frontend assets natively).

When it finishes:
```bash
sudo docker compose ps
sudo docker compose logs --tail 50
```

You should see:
- `algotrader` container status: `Up X seconds (healthy)`
- Logs ending with: `Halt-resume monitor enabled` and `Streaming... waiting for signals`

---

## Phase 6: Open dashboard + sanity check (2 min)

In a browser:
```
http://<NEW_PUBLIC_IP>
```

You should see the AlgoTrader dashboard with the **$27K paper account** balance.

---

## Phase 7: Set up auto-deploy cron (5 min)

```bash
crontab -e
```

Add this line at the bottom:
```
*/5 * * * * cd /home/ubuntu/algo-trading && git fetch -q origin main && if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then git pull -q origin main && sudo docker compose up -d --build >> logs/deploy.log 2>&1; fi
```

Same as old AWS cron — pulls from GitHub every 5 min and rebuilds if there's a new commit.

(Note: We never fixed the auth on the old AWS, but the repo is public so `git pull`
works without auth. If you've made it private, we'll need to set up a deploy key.)

---

## Phase 8: Verify trading (5 min)

```bash
# Watch live logs for ~2 minutes to confirm scanner + streamer + halt monitor are alive
sudo docker compose logs -f --tail 30
```

Look for:
- `Scanner` activity (or "Waiting for 7:00 ET" if market closed)
- `live.streamer` WebSocket connection
- `live.halt_monitor` polling

To smoke-test order execution without market action, you can manually submit a paper order via Alpaca dashboard and see if the bot recognizes it.

---

## Phase 9: (Optional) Set up reserved public IP

If you want a fixed IP that survives instance restarts:

1. Networking → Reserved Public IPs → Reserve
2. Attach to your `algotrader` instance's primary VNIC

This is the OCI equivalent of AWS Elastic IP. **Free** as long as it's attached
to a running instance.

---

## What's lost in the migration

| Item | Status |
|---|---|
| `fills_calibration.csv` (slippage Stage 2 data) | ❌ Lost — restarts from zero |
| Past `logs/*_trades.json` | ❌ Lost — but trade history is on Alpaca dashboard |
| Old AWS Elastic IP `52.2.131.240` | ❌ Lost — will get a new IP |
| Old SSH key `trading-key.pem` | ❌ Useless on new account |

| Item | Recovered |
|---|---|
| Trading config (`trial_6_extracted.json`) | ✅ In git |
| Alpaca API keys | ✅ In local `config/settings.py` + this runbook |
| Strategy code | ✅ In git |
| Open positions (if any in old account) | ✅ Visible on Alpaca dashboard — bot will recover them on first start via `RECOVERY` logic in `live/main.py` |

---

## After migration: update local references

Update these locally so they point at the new IP:

1. **README.md**: replace `52.2.131.240` with the new Oracle IP everywhere
2. **Memory** (`memory/MEMORY.md`): update the IP reference if any

---

## If something goes wrong

- **Can't SSH**: confirm Security List opened port 22, and iptables not blocking
- **`docker compose up` fails on build**: check `Dockerfile` ARM compatibility — base images should be multi-arch, but if anything pins x86 explicitly, we'd need to patch
- **Container starts but bot doesn't trade**: check `.env` for the right keys; check `docker compose logs algotrader | grep -i error`
- **Out of ARM capacity**: try a different availability domain in same region, or retry after a few hours. **Do not pay for a paid shape** — that's the whole point of moving here.

---

## Total estimated time

| Phase | Time |
|---|---|
| Account signup + VM provision | 30 min (mostly waiting) |
| SSH + Docker install | 15 min |
| Repo clone + config | 10 min |
| First build + start | 10 min |
| Dashboard check + cron | 10 min |
| **Total** | **~75 min** of active work |

You'll be back online with the bot trading on the new Oracle host within
about an hour of starting.
