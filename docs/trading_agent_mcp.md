# Trading Agent: Alpaca MCP Control Plane

## Decision

Alpaca MCP v2 is part of the trading-agent architecture as an **operator/control plane**, not as the live signal-data path.

The production boundary is:

```text
Tradier real-time stream
        |
        v
deterministic strategy engine
        |
        v
portfolio / risk engine
        |
        v
deterministic execution engine
        |
        v
Alpaca Trading API

Historical / backfill / validation:
Alpaca SIP historical APIs -> canonical CSV/CSV.GZ research store

AI operator path:
Trading agent -> Alpaca MCP v2 -> Alpaca account/data APIs
```

## Why this split

The live strategies depend on low-latency, full-market data. Tradier remains the primary real-time stream used for signal generation and participation/liquidity calculations.

The free Alpaca account can still be useful for historical SIP once data is old enough, but Alpaca MCP does not remove Alpaca account data entitlements or the free-tier SIP delay. Therefore MCP must not become a hidden dependency for live signal generation.

The deterministic engine remains authoritative. The LLM may inspect, explain, monitor, diagnose, and propose actions; it does not get to invent entries/exits or silently alter strategy parameters during the session.

## Initial MCP permissions

The sidecar starts with read-oriented toolsets only:

```text
account,assets,stock-data,news,corporate-actions
```

The following are intentionally disabled by default:

```text
trading
locates
```

This lets the agent answer questions such as:

- What is the Alpaca account equity, cash, and buying power?
- What positions or broker-side discrepancies exist?
- Pull historical bars or a stock snapshot for diagnosis.
- Check corporate actions that could explain a price discontinuity.
- Pull Alpaca news for a ticker as contextual metadata.
- Compare broker/account state against the deterministic engine's persisted state.

It does **not** let the model place or cancel orders through MCP by default.

## Execution policy

Normal strategy execution continues to use the existing direct broker adapter. MCP is not inserted between the strategy/risk engine and Alpaca.

That is deliberate:

1. strategy decisions remain reproducible and testable;
2. execution latency is not coupled to an LLM/tool round trip;
3. risk checks cannot be skipped because an agent chose a different tool sequence;
4. restart/reconciliation logic remains deterministic;
5. paper/live parity remains measurable.

If the MCP `trading` toolset is enabled later, it should be restricted to operator workflows such as an explicitly approved manual flatten/cancel action or controlled paper-trading experiments. It should not become the strategy execution path.

## Tradier remains primary live data

The trading-agent capability manifest explicitly records:

```text
primary_live_feed = Tradier
historical_provider = Alpaca SIP
execution_broker = Alpaca direct API
operator_control_plane = Alpaca MCP v2
```

This avoids a dangerous ambiguity where the agent could treat Alpaca's free real-time IEX or delayed SIP data as interchangeable with the Tradier stream.

## Sidecar deployment

An optional image is provided at:

```text
deploy/alpaca-mcp/Dockerfile
```

The Compose service is behind the `agent` profile and binds only to localhost on the host. Start it with:

```bash
docker compose --profile agent up -d --build alpaca-mcp
```

Default environment mapping:

```text
ALPACA_API_KEY       <- existing Alpaca key
ALPACA_SECRET_KEY    <- existing ALPACA_API_SECRET
ALPACA_PAPER_TRADE   <- existing ALPACA_PAPER
ALPACA_TOOLSETS      <- ALPACA_MCP_TOOLSETS
```

Default toolsets are read-oriented. No new secrets are committed.

## Remote ChatGPT access

Alpaca's official MCP server supports streamable HTTP, but it does **not** provide the remote OAuth/authentication layer required to expose a broker-connected MCP service safely on the public internet.

Therefore this repo does not publish port 8001 publicly.

For a future ChatGPT-connected deployment, place a standards-based authenticated MCP gateway in front of the private sidecar and set FastMCP's allowed-host configuration to the exact public MCP hostname. Do not solve this by exposing `0.0.0.0:8001` directly.

Until that gateway exists, the sidecar can be used by trusted local/internal agent processes or through a secured tunnel.

## Safety progression

### Stage A — current default

- MCP sidecar disabled unless the `agent` Compose profile is selected.
- Paper account by default.
- Read-oriented MCP toolsets only.
- Tradier is the live strategy feed.
- Direct Alpaca broker adapter remains the only automated execution path.

### Stage B — operator actions in paper

Only after audit logging and approval controls are in place:

- enable `trading` in MCP for paper;
- require explicit operator approval for any order mutation;
- tag every agent-originated order with a distinct client-order namespace;
- reconcile MCP-originated actions back into the same event/audit store.

### Stage C — live operational controls

Only after paper/shadow parity has been demonstrated:

- use live credentials in a separately scoped deployment;
- keep strategy-generated orders on the deterministic execution path;
- permit only narrowly defined operator actions through the agent;
- retain broker-side kill switches and position reconciliation independent of the LLM.

## Locates

Alpaca MCP exposes a `locates` toolset for supported accounts, but locates are not available in Alpaca paper trading. If/when the RTH short system moves to a compatible live account, locate availability/cost can be treated as an execution constraint or monitoring input.

It should not be used as a historical feature unless the historical data is truly available at the relevant decision timestamp.

## Source

The integration targets Alpaca MCP Server v2. Its official documentation states that v2 uses FastMCP/OpenAPI, supports server-side toolset filtering via `ALPACA_TOOLSETS`, supports streamable HTTP, and warns that the package does not configure remote MCP authentication.

Reference: https://github.com/alpacahq/alpaca-mcp-server
