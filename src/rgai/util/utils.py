import asyncio
import yaml
from pathlib import Path
from typing import Any, Dict, List

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

async def get_tools_by_server_name(mcpServers) -> Dict[str, List[BaseTool]]:
    client = MultiServerMCPClient(
      connections = mcpServers
    )
    tools_by_server = {}
    for server_name in mcpServers.keys():
        tools = await client.get_tools(server_name=server_name)
        tools_by_server[server_name] = tools
    return tools_by_server

def get_tools(mcpServers):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, so create and run a new one
        # print("No running loop found, creating a new one.")
        return asyncio.run(get_tools_by_server_name(mcpServers))
    else:
        # Loop is running, use run_until_complete
        # print("Running loop found, using run_until_complete.")
        return loop.run_until_complete(get_tools_by_server_name(mcpServers))

# Helper function to recursively extract 'content' from all AIMessage instances
def extract_ai_message_content(stream) -> List[str]:
    # Extract AIMessage contents into an array as {key: value}
    ai_message_contents = []

    for key, value in stream.items():
            # We expect a 'messages' key
            if value is None:
                continue
            messages = value['messages']
            if isinstance(messages, list):  # Skip ToolMessages or others
                continue
            ai_message_contents.append((key, messages.content)) # Assuming messages is an AIMessage
            # print("ai_message_contents...\n")
            # print(ai_message_contents)
            # if isinstance(messages, dict):  # AIMessage will typically be a dict
            #     ai_message_contents.append((key, messages.content))
            #     # if 'content' in messages:
            #     #     ai_message_contents.append({key: messages['content']})
            # elif isinstance(messages, list):  # Skip ToolMessages or others
            #     continue
        
    return ai_message_contents

def parse_messages(result: dict):
    data = {
        "human": [],
        "ai": [],
        "tools": [],
    }

    for msg in result.get("messages", []):
        if hasattr(msg, "name"):  # ToolMessage (e.g., name='write_file')
            data["tools"].append({
                "tool_name": getattr(msg, "name", None),
                "tool_call_id": getattr(msg, "tool_call_id", None),
                "content": getattr(msg, "content", None)
            })
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            # AIMessage with tool calls (the reasoning + tool actions)
            calls = []
            for call in msg.tool_calls:
                calls.append({
                    "name": call["name"],
                    "args": call["args"]
                })
            data["ai"].append({
                "content": getattr(msg, "content", None),
                "tool_calls": calls
            })
        else:
            # HumanMessage or simple AIMessage
            msg_type = type(msg).__name__
            entry = {"type": msg_type, "content": getattr(msg, "content", None)}
            if msg_type.lower().startswith("human"):
                data["human"].append(entry)
            else:
                data["ai"].append(entry)
    return data

def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_field_config(field_config_path: str | Path) -> Dict[str, Dict[str, Any]]:
    cfg = load_yaml(field_config_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"field_config must be a mapping; got {type(cfg)}")
    return cfg
