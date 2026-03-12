# Stage 1: Build React frontend
FROM node:22-slim AS frontend-build
WORKDIR /app/dashboard/frontend
COPY dashboard/frontend/package*.json ./
RUN npm ci --production=false
COPY dashboard/frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.11-slim AS runtime
WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed for live trading
COPY config/ ./config/
COPY live/ ./live/
COPY dashboard/ ./dashboard/
COPY *.py ./

# Copy built frontend (overwrite source with built assets)
COPY --from=frontend-build /app/dashboard/frontend/dist /app/dashboard/frontend/dist

# Create logs directory
RUN mkdir -p /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/account')" || exit 1

EXPOSE 8000

# Default: live trading + dashboard
CMD ["python", "-m", "live.main"]
