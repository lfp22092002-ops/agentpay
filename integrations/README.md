# AgentPay Framework Integrations

Drop-in tool wrappers for popular agent frameworks.

## LangChain

```python
from agentpay.integrations.langchain import get_agentpay_tools

tools = get_agentpay_tools(api_key="ap_xxxx...")
# Use with any LangChain agent: create_react_agent, create_openai_functions_agent, etc.
```

## CrewAI

```python
from agentpay.integrations.crewai import get_agentpay_tools
from crewai import Agent

tools = get_agentpay_tools(api_key="ap_xxxx...")
agent = Agent(role="Buyer", tools=tools, goal="Purchase compute resources")
```

## Smolagents (HuggingFace)

```python
from agentpay.integrations.smolagents import get_agentpay_tools
from smolagents import CodeAgent, HfApiModel

tools = get_agentpay_tools(api_key="ap_xxxx...")
agent = CodeAgent(tools=tools, model=HfApiModel())
```

## Google ADK (Agent Development Kit)

```python
from integrations.google_adk import get_agentpay_tools
from google.adk.agents import Agent

tools = get_agentpay_tools(api_key="ap_xxxx...")
agent = Agent(name="buyer", model="gemini-2.0-flash", tools=tools)
```

## Available Tools

| Tool | Description | Frameworks |
|------|-------------|------------|
| `agentpay_balance` | Check wallet balance (USD, spent, received) | All |
| `agentpay_spend` | Spend funds with description and category | All |
| `agentpay_transfer` | Transfer USD to another agent | All |
| `agentpay_transactions` | List recent transactions with filters | All |
| `agentpay_x402_pay` | Pay for HTTP 402 resources (x402 protocol) | All |
| `agentpay_identity` | Get agent identity and trust score | LangChain |

## MCP Server

For frameworks that support MCP natively (Claude Code, Cursor, OpenClaw, OpenAI Agents SDK):

```json
{
  "mcpServers": {
    "agentpay": {
      "url": "https://leofundmybot.dev/mcp",
      "headers": { "x-api-key": "YOUR_API_KEY" }
    }
  }
}
```
