# AWS Migration Runbook (new account)

Migrating the algo-trading bot from the old AWS account (suspended) to a fresh
AWS account.

**Target**: EC2 instance, Ubuntu 22.04, us-east-1 (N. Virginia — closest to
Alpaca NJ), Docker stack identical to before.

---

## Phase 1: Create the new AWS account (~15 min)

1. Go to <https://aws.amazon.com/>, click "Create an AWS Account"
2. Use a **fresh email** (the old account's email is permanently associated with the suspended one)
3. Pick "Personal" account type
4. Credit/debit card required for billing verification
5. Phone verification (SMS or voice call)
6. Pick the **Basic Support — Free** plan
7. Wait for "Account activation complete" email (usually <10 min)

---

## Phase 2: Pick a region + provision the EC2 instance (~10 min)

### 2.1 Switch region to `us-east-1` (N. Virginia)

Top-right of AWS Console → region dropdown → **US East (N. Virginia)** —
closest to Alpaca NJ datacenter, lowest live-trading latency.

### 2.2 Launch instance

EC2 → Instances → **Launch instances**

| Setting | Value |
|---|---|
| Name | `algotrader` |
| AMI | **Ubuntu Server 22.04 LTS** (x86_64) — Free tier eligible |
| Instance type | `t3.micro` if staying free-tier, or `t3.small` if you want headroom |
| Key pair | **Create new** — call it `trading-key-v2`, type `RSA`, format `.pem`. Download immediately — you can't redownload later |
| Network → VPC | Default |
| Network → Subnet | Default (any AZ in us-east-1) |
| **Auto-assign public IP** | **Enable** |
| Security group | Create new → call it `algotrader-sg` |

Security group rules — **move-anywhere setup** (SSH key is your security, not IP allow-list):
- Inbound: **Custom TCP 2222** from `0.0.0.0/0` (SSH on a non-standard port)
- Inbound: **HTTP (80)** from `0.0.0.0/0` (dashboard access)
- Inbound: **Custom TCP 8000** from `0.0.0.0/0` (optional, direct dashboard)
- All outbound: allow (default)

⚠️ **Do NOT add a port 22 rule.** We're moving SSH to port 2222 in Phase 5
to eliminate ~95% of brute-force scanner noise. The SSH key (`.pem`) is the
actual security barrier; key-only auth is strong even with an open IP rule.

If you'd prefer the locked-down "My IP" approach instead (more secure but
breaks when your home IP changes), use this rule instead:
- Inbound: **SSH (22)** from `My IP` — and skip the port-2222 change in Phase 5

Storage:
- **20 GB gp2** (NOT gp3 — gp3 isn't free-tier-eligible)

Click **Launch instance**. Wait ~1 min for it to show "Running" + status checks `2/2 passed`.

### 2.3 Allocate + attach an Elastic IP

EC2 → Elastic IPs → **Allocate Elastic IP address** → Allocate
→ Right-click the new EIP → **Associate Elastic IP address** → select your `algotrader` instance → Associate

Note the EIP address — that's your bot's permanent public IP.

⚠️ **The EIP is only free while attached to a running instance.** Never stop the instance without first re-associating the EIP — otherwise you'll be billed ~$3.60/month for the unattached EIP.

---

## Phase 3: First-time SSH access on port 22 (~5 min)

**Note**: the first SSH happens on port 22 (AWS default) — we'll switch to
2222 in Phase 5. Temporarily add a port-22 SG rule from your current IP so
you can get in for the initial setup. **Delete that rule after Phase 5**.

In AWS Console: EC2 → Security Groups → `algotrader-sg` → Edit inbound rules
→ Add a temporary rule:
- **SSH (22)** from `My IP`
- Save

Then SSH in:

```bash
chmod 600 trading-key-v2.pem
ssh -i trading-key-v2.pem ubuntu@<NEW_EIP>
```

You should land at `ubuntu@ip-...$`. First-time prompts:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y htop ncdu fail2ban
```

(`fail2ban` auto-bans IPs after repeated failed login attempts — defense in
depth on top of key-only auth.)

---

## Phase 4: Add swap (optional but recommended for t3.micro)

Only needed if you used t3.micro (1 GB RAM). Skip for t3.small (2 GB RAM).

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

Verifies you now have 1 GB RAM + 2 GB swap = effective 3 GB.

---

## Phase 5: Install Docker + lock down SSH (~10 min)

### 5.1 Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
```

### 5.2 Move SSH to port 2222 (move-anywhere setup)

While still in the existing port-22 SSH session:

```bash
sudo sed -i 's/^#Port 22/Port 2222/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```

⚠️ **Do NOT log out yet.** First verify port 2222 works from a SECOND
terminal on your laptop:

```bash
# In a NEW terminal locally:
ssh -i trading-key-v2.pem -p 2222 ubuntu@<NEW_EIP>
```

If port 2222 works, you can safely close the original port-22 session.
If it doesn't work, fix the issue from the still-open port-22 session before
closing it (otherwise you'll be locked out and need to relaunch via
EC2 Instance Connect through the AWS Console).

### 5.3 Remove the temporary port-22 SG rule

Now that 2222 works, go back to AWS Console:
- EC2 → Security Groups → `algotrader-sg` → Edit inbound rules
- **Delete** the port-22 rule from Phase 3
- Save

From now on, all SSH uses `-p 2222`:

```bash
ssh -i trading-key-v2.pem -p 2222 ubuntu@<NEW_EIP>
docker ps                       # confirms Docker works without sudo
```

---

## Phase 6: Clone the repo + configure `.env` (~5 min)

```bash
cd ~
git clone https://github.com/korirtheo/algo-trading.git
cd algo-trading

cat > .env <<'EOF'
ALPACA_API_KEY=PKF56A65KVUCGU4DBDPIYRKHNC
ALPACA_API_SECRET=7pmp8Nk3dkZqWweqkAsB9Rkg6QDbBn4YrKfxv6V5Q14p
ALPACA_PAPER=true
LIVE_PARAMS_PATH=config/trial_6_extracted.json
EOF
chmod 600 .env
```

---

## Phase 7: Build + start (~5-10 min)

```bash
sudo docker compose up -d --build
```

Build takes ~5-8 min (Node frontend + Python deps). When done:

```bash
sudo docker compose ps
sudo docker compose logs --tail 50
```

Expected: `algotrader  Up X seconds (healthy)` + logs ending with
`Halt-resume monitor enabled` and `Streaming... waiting for signals`.

---

## Phase 8: Dashboard sanity check (~2 min)

In a browser:
```
http://<NEW_EIP>
```

Should show the dashboard with **$27K paper balance**.

---

## Phase 9: Auto-deploy cron (~5 min)

```bash
crontab -e
```

Add this line:
```
*/5 * * * * cd /home/ubuntu/algo-trading && git fetch -q origin main && if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then git pull -q origin main && sudo docker compose up -d --build >> logs/deploy.log 2>&1; fi
```

Pulls from GitHub every 5 min, rebuilds Docker only when there's a new commit.

---

## Phase 10: CRITICAL — Set up billing alert (~5 min)

This catches any charge instantly so you can shut it down before damage:

1. Billing & Cost Management → **Budgets** → Create budget
2. Use a template → **Zero spend budget**
3. Email recipient: your email
4. Threshold: $0.01 (catches any non-free usage instantly)
5. Create

You'll get an email the moment AWS bills you anything.

Also enable **billing alarms in CloudWatch** (one-time toggle):
Billing → Billing preferences → Receive billing alerts → **Enabled**

---

## What needs updating locally after migration

After the new IP is live:

1. **README.md**: replace `52.2.131.240` references with `<NEW_EIP>` — there are 2-3 places
2. **Memory** (`memory/MEMORY.md`): if any IP reference, update

---

## What's lost in the migration

| Item | Status |
|---|---|
| `fills_calibration.csv` (Stage 2 slippage data) | ❌ Lost — can't access old account |
| Old trade logs | ❌ Lost — same reason |
| Old EIP `52.2.131.240` | ❌ Belongs to old account |
| Old SSH key `trading-key.pem` | ❌ Useless |

| Item | Recovered |
|---|---|
| All source code | ✅ In git |
| Alpaca keys | ✅ In `.env` setup above |
| Trial #6 params (live config) | ✅ In `config/trial_6_extracted.json` |
| Open positions (if any in $27K account) | ✅ Bot's `RECOVERY` logic in `live/main.py` picks them up automatically on first start |

---

## Estimated total time

| Phase | Time |
|---|---|
| Account creation + activation | 15-30 min |
| EC2 launch + EIP | 10 min |
| SSH + setup | 10 min |
| Docker install + clone | 10 min |
| First build | 10 min |
| Dashboard + cron + billing alarm | 15 min |
| **Total** | **~60-90 min** active work |

---

## Quick troubleshooting

| Symptom | Fix |
|---|---|
| Can't SSH | Check SG inbound rule allows port **2222** from `0.0.0.0/0`. If still stuck, use EC2 Instance Connect from the AWS Console as a fallback (Connect button on the instance → EC2 Instance Connect → opens an in-browser shell). From there you can fix sshd_config or SG rules. |
| `docker compose` says permission denied | You forgot to log out + back in after `usermod -aG docker`. Reconnect. |
| `Out of memory` during build | Add the swap file (Phase 4) and rebuild. |
| Container starts but no trades | `docker compose logs algotrader` — check for `Account: cash=$27,000` confirming keys loaded. If not, recheck `.env`. |
| Dashboard 404/timeout | Check SG inbound port 80 rule. Check container is `Up (healthy)`. |
| Billing email arrives | Stop the instance IMMEDIATELY (`sudo docker compose down` then in AWS Console stop the instance). Investigate what's costing money before restarting. |

---

## What I'd do differently this time

The old AWS account got billed probably because:
1. **t2/t3.micro RAM too small** → may have moved to t3.small ($15/mo)
2. **EIP detached at some point during the 2-month outage** → ~$7 in idle-EIP fees
3. **Possibly upgraded EBS to gp3** → ~$0.20/GB-month above free tier

Following Phase 4 (swap) and the "never detach EIP" rule + the billing alarm in Phase 10 should keep this account free.
