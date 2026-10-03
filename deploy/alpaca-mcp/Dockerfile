FROM python:3.11-slim

RUN pip install --no-cache-dir "alpaca-mcp-server>=2,<3"

EXPOSE 8001

# Streamable HTTP is used only on the private/local side of the deployment.
# Do not expose this container directly to the public internet: Alpaca's MCP
# package does not provide remote MCP OAuth/authentication.
CMD ["alpaca-mcp-server", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8001"]
