# Using with LangGraph StateGraph

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.tools import tool
from langchain_core.runnables import Runnable, RunnableConfig
from typing import TypedDict, Literal
import asyncio
import json

@tool
def add(a: float, b: float):
    """Add two numbers."""
    return a + b

@tool
def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b

primary_agent_prompt = """You are a primary assistant tasked with managing a conversation between following workers.
                    - devops_assistant: Handles all tasks related to devopss.
                    - jira_assistant: Handles all tasks related to jira.
                    - math_assistant: Handles all tasks related to math.
                    Given the following user request, respond with the worker to act next. Each worker will perform a
                    task and respond with their results and status. When finished, respond with FINISH.
                    UTILIZE last conversation to assess if the conversation should end you answered the query, then route to FINISH """

jira_agent_prompt = """You are specialized agent to provide information related to jira from a jira project."""

devops_agent_prompt = """You are specialized agent to provide devops related information."""

math_agent_prompt = "You are a math assistant. You can perform basic arithmetic operations like addition, multiplication, and division. Always provide the correct answer to the user query."

MEMBERS = ["devops_assistant", "jira_assistant", "math_assistant"]
OPTIONS = MEMBERS + ["FINISH"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {MEMBERS}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."

    "Guidelines:\n"
    "1. Always check the last message in the conversation to determine if the task has been completed.\n"
    "2. If you already have the final answer or outcome, return 'FINISH'.\n"
)

# from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
# llm = ChatOCIGenAI(
#             model_id="cohere.command-r-08-2024",
#             service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
#             compartment_id="ocid1.compartment.oc1..xxxx",
#             auth_type="API_KEY",
#             auth_profile="xxxxx",
#             model_kwargs={"temperature": 0, "top_p": 0.75, "max_tokens": 512}
#         )

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:14b")

connections = {
    "mcp-devops": {
      "timeout": 60,
      "transport": "sse",
      "url": "http://localhost:8081/sse"
    },
    "mcp-atlassian": {
        "command": "docker",
        "args": [
          "run",
          "-i",
          "--rm",
          "-e", "CONFLUENCE_URL",
          "-e", "CONFLUENCE_USERNAME",
          "-e", "CONFLUENCE_API_TOKEN",
          "-e", "JIRA_URL",
          "-e", "JIRA_USERNAME",
          "-e", "JIRA_API_TOKEN",
          "ghcr.io/sooperset/mcp-atlassian:latest"
        ],
        "env": {
          "CONFLUENCE_URL": "https://your-company.atlassian.net/wiki",
          "CONFLUENCE_USERNAME": "your.email@company.com",
          "CONFLUENCE_API_TOKEN": "your_confluence_api_token",
          "JIRA_URL": "https://your-company.atlassian.net",
          "JIRA_USERNAME": "your.email@company.com",
          "JIRA_API_TOKEN": "your_jira_api_token"
        }
    },
    "mcp-math": {
        "command": "python",
        # Make sure to update to the full absolute path to your mcp_server.py file
        "args": ["mcp_server.py"],
        "transport": "stdio",
    },
    # "weather": {
    #     # make sure you start your weather server on port 8000; $python mcp_server_weather.py
    #     "url": "http://localhost:8000/sse",
    #     "transport": "sse",
    # }
}

def chatbot(state: MessagesState, tools: list[BaseTool]):
            response = llm.bind_tools(tools).invoke(state["messages"])
            return {"messages": response}

def route_to_assistant(state: MessagesState,):
    route = tools_condition(state)
    print("route:" + route)
    if route == "__end__":
        return END
    tool_calls = state["messages"][-1].tool_calls
    # print(state["messages"])
    # print("####tool_calls####")
    # print(tool_calls)
    if tool_calls:
        if tool_calls[0]["name"] == "devops_subagent":
            print(tool_calls[0]["name"])
            return "devops_assistant"
        elif tool_calls[0]["name"] == "jira_subagent":
            print(tool_calls[0]["name"])
            return "jira_assistant"
        else:
            print(tool_calls[0]["name"])
            return "math_assistant"
    raise ValueError("Invalid route")

def _create_agent(llm, tools, agent_prompt):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
            "system", agent_prompt
            ),
            ("placeholder", "{messages}"),
        ]
    )
    agent_runnable = prompt_template | llm.bind_tools(tools)
    return agent_runnable

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: MessagesState, config: RunnableConfig):
        # do some processing if needed
        #
        result = self.runnable.invoke(state)
        return {"messages": result}

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*OPTIONS]
    
def supervisor_node(state: MessagesState) -> Router:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response : Router = llm.with_structured_output(Router).invoke(messages)
    print("\nprinting full response from supervisor node...")
    print(response)
    # return response
    #return {"messages": [AIMessage(content=json.dumps(response))]}

@tool
def devops_subagent(request: str) -> str:
    """Provide devops related information like get logs, metrics, etc.

    Input: Natural language request for devops information like logs, metrics, etc.(e.g., 'download logs for the opc-request-id')
    """

    return "devops_assistant"

@tool
def jira_subagent(request: str) -> str:
    """Provide information related to jira from a jira project queue.

    Input: Natural language request for jira / issue information like comments, description, etc. (e.g., 'Get the details of jira id RG-3552 the RG project queue.')
    """

    return "jira_assistant"

@tool
def math_subagent(request: str) -> str:
    """Do mathematical calculations.

    Use this tool when the user wants to add, delete, mulitiply, or divide numbers.
    
    Input: Natural language computation request (e.g., 'calculate the result of 3 + 5')
    """
    return "math_assistant"

async def my_async_function():
    client = MultiServerMCPClient(
        connections
    )
    mcp_devops_tools = await client.get_tools(server_name="mcp-devops")
    mcp_atlassian_tools = await client.get_tools(server_name="mcp-atlassian")
    mcp_math_tools = await client.get_tools(server_name="mcp-math")

    devops_assistant = _create_agent(
                    llm=llm,
                    tools= mcp_devops_tools,
                    agent_prompt=devops_agent_prompt
    )
    devops_tool_node = ToolNode(mcp_devops_tools)

    jira_assistant = _create_agent(
                    llm=llm,
                    tools= mcp_atlassian_tools,
                    agent_prompt=jira_agent_prompt
    )
    jira_tool_node = ToolNode(mcp_atlassian_tools)

    math_assistant = _create_agent(
                    llm=llm,
                    tools= mcp_math_tools,
                    agent_prompt=math_agent_prompt
    )
    math_tool_node = ToolNode(mcp_math_tools)

    primary_tools = [devops_subagent, jira_subagent, math_subagent]
    primary_assistant = _create_agent(
                        llm=llm,
                        tools= primary_tools,
                        agent_prompt=primary_agent_prompt
    )

    builder = StateGraph(MessagesState)
    
    builder.add_node("primary_assistant", Assistant(primary_assistant))
    # builder.add_node("primary_assistant", supervisor_node)
    builder.add_node("devops_assistant", Assistant(devops_assistant))
    builder.add_node("jira_assistant", Assistant(jira_assistant))
    builder.add_node("math_assistant", Assistant(math_assistant))

    builder.add_edge(START, "primary_assistant")

    builder.add_conditional_edges(
        "primary_assistant",
        route_to_assistant,
        [
            "devops_assistant",
            "jira_assistant",
            "math_assistant",
            END,
        ],
        
        # lambda x: x["messages"][-1].tool_calls[0]["name"] if x["messages"] and x["messages"][-1].tool_calls else "FINISH",
        # {
        #     "devops_subagent" :  "devops_assistant",
        #     "jira_subagent" : "jira_assistant",
        #     "math_subagent" : "math_assistant",
        #     "FINISH": END,
        # },

        # lambda router_output: json.loads(router_output["messages"][-1].content)["next"],
        # {
        #     "devops_assistant": "devops_assistant",
        #     "jira_assistant": "jira_assistant",
        #     "math_assistant": "math_assistant",
        #     "FINISH": END,
        # },
    )

    # Add the tool_node as a node
    builder.add_node("jira_tools_executor", jira_tool_node)
    builder.add_edge("jira_assistant", "jira_tools_executor")
    builder.add_edge("jira_tools_executor", "primary_assistant") # report back to primary_assistant

    builder.add_node("devops_tools_executor", devops_tool_node)
    builder.add_edge("devops_assistant", "devops_tools_executor")
    builder.add_edge("devops_tools_executor", "primary_assistant") # report back to primary_assistant

    builder.add_node("math_tools_executor", math_tool_node)
    builder.add_edge("math_assistant", "math_tools_executor")
    builder.add_edge("math_tools_executor", "primary_assistant") # report back to primary_assistant

    graph = builder.compile()
    #print(builder.get_graph().draw_mermaid())

    # from langgraph.graph import get_graph_viz
    # graph_viz = get_graph_viz(builder)
    # with open("my_graph.png", "wb") as f:
    #     f.write(graph_viz)

    # agent_response = await graph.ainvoke({"messages": "Get the details of jira id RG-3552 the RG project queue."})
    # agent_response = await graph.ainvoke({"messages": "calculate the result of (3 + 5) x 7"})
    # print(agent_response)

    #user_input = {"messages": "Get the details of jira id RG-3552 the RG project queue. And summarie its last comment."}
    user_input = {"messages": "calculate the result of 3 + 5 using the math tool."}
    # user_input = {
    #         "role": "human",
    #         "content": "calculate the result of 3 + 5 using the math tool"
    # }
    #user_input = {"messages": "Who is the prime minister of India?"}

    # agent_response = await graph.ainvoke({"messages": user_input})
    # print(agent_response)

    async for step in graph.astream(user_input):
        # 'step' will contain information about the current node's output
        # You can inspect 'step' to understand the execution flow
        print(step)

if __name__ == "__main__":
    asyncio.run(my_async_function())
