import json
from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain_core.tools import tool

from aio.core.llm_provider import LLMFactory
from aio.util.utils import get_tools

# ===================================== #
#               SUBAGENTS CONFIG
# ===================================== #

with open('config/subagents_config.json', 'r', encoding='utf-8') as f:
    subagents_config = json.load(f)

SUBAGENTS = subagents_config["subagents"]
SUBAGENTS_CAPABILITIES = subagents_config["subagents_capabilities"]
SUBAGENTS_PROMPTS = subagents_config["subagents_prompts"]
SUPERVISOR_PROMPT = subagents_config["supervisor_prompt"]
MCP_SERVERS = subagents_config["mcpServers"]
MODEL_PROVIDER = subagents_config["model_provider"]

# ===================================== #
#               MODEL
# ===================================== #

# The model is the reasoning engine of your agent.
llm = LLMFactory.create_llm(MODEL_PROVIDER)

# ===================================== #
#               TOOLS
# ===================================== #

# Tools give agents the ability to take actions.
tools = get_tools(MCP_SERVERS)

# ===================================== #
#     SUBAGENTS AS TOOLS
# ===================================== #

def _create_subagent_as_tool(subagent: str = None):
    """Create tools from agents"""

    # Function to export an agent as a tool
    def export_agent_as_tool(agent, tool_name: str, tool_description: str):
        """Create a tool from an agent"""
        @tool(name_or_callable=tool_name, description=tool_description)
        async def agent_as_tool(request: str) -> str:
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": request}]
            })
            return result["messages"][-1].text
        return agent_as_tool

    # create_agent builds a graph-based agent runtime using LangGraph. 
    # A graph consists of nodes (steps) and edges (connections) that define how your agent processes information. 
    # The agent moves through this graph, executing nodes like the model node (which calls the model), the tools node (which executes tools), or middleware.
    # Agents combine language models with tools to create systems that can reason about tasks, decide which tools to use and iteratively work towards solutions.
    # An LLM Agent runs tools in a loop to achieve a goal.
    agent = create_agent(
        model=llm,
        tools=tools[f"{subagent}_mcp"] if f"{subagent}_mcp" in tools.keys() else [],
        system_prompt=SUBAGENTS_PROMPTS[subagent]
    )
    agent_as_tool =  export_agent_as_tool(
        agent=agent,
        tool_name=f"{subagent}_agent_tool",
        tool_description=SUBAGENTS_CAPABILITIES[subagent]
    )

    return agent_as_tool

# ===================================== #
#              SUPERVISOR
# ===================================== #

def create_supervisor():
    """Create the supervisor that manages the agents"""
    supervisor = create_agent(
        model=llm,
        tools=[_create_subagent_as_tool(subagent) for subagent in SUBAGENTS],
        system_prompt=SUPERVISOR_PROMPT
    )
    return supervisor

# ============================= #
#           SUBAGENTS
# ============================= #

# Define the subagent configurations
def _create_subagent(subagent: str = None):
    """Create subagents"""
    subagent_config = {
        "name": f"{subagent}_subagent",
        "description": SUBAGENTS_CAPABILITIES[subagent],
        "system_prompt": SUBAGENTS_PROMPTS[subagent],
        "tools": tools[f"{subagent}_mcp"] if f"{subagent}_mcp" in tools.keys() else [],
    }

    return subagent_config

# ===================================== #
#            DEEPAGENT
# ===================================== #

def create_deepagent():
    """Create the deepagent that manages the agents"""
    deepagent = create_deep_agent(
        model=llm,
        tools=[],
        system_prompt=SUPERVISOR_PROMPT,
        subagents=[_create_subagent(subagent) for subagent in SUBAGENTS],
    )
    return deepagent