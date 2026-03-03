import os
from getpass import getpass
os.environ['TAVILY_API_KEY'] = getpass('Enter Tavily API Key: ')

import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient()

def internet_search(
query: str,
max_results: int = 5,
topic: Literal["general", "news", "finance"] = "general",
include_raw_content: bool = False,
):
    """Run a web search"""
    search_docs = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs

sub_research_prompt = """
You are a specialized AI policy researcher.
Conduct in-depth research on government policies, global regulations, and ethical frameworks related to artificial intelligence.

Your answer should:
- Provide key updates and trends
- Include relevant sources and laws (e.g., EU AI Act, U.S. Executive Orders)
- Compare global approaches when relevant
- Be written in clear, professional language

Only your FINAL message will be passed back to the main agent.
"""

research_sub_agent = {
    "name": "policy-research-agent",
    "description": "Used to research specific AI policy and regulation questions in depth.",
    "system_prompt": sub_research_prompt,
    "tools": [internet_search],
}

sub_critique_prompt = """
You are a policy editor reviewing a report on AI governance.
Check the report at `final_report.md` and the question at `question.txt`.

Focus on:
- Accuracy and completeness of legal information
- Proper citation of policy documents
- Balanced analysis of regional differences
- Clarity and neutrality of tone

Provide constructive feedback, but do NOT modify the report directly.
"""

critique_sub_agent = {
    "name": "policy-critique-agent",
    "description": "Critiques AI policy research reports for completeness, clarity, and accuracy.",
    "system_prompt": sub_critique_prompt,
}

policy_research_instructions = """
You are an expert AI policy researcher and analyst.
Your job is to investigate questions related to global AI regulation, ethics, and governance frameworks.

1️Save the user's question to `question.txt`
2️ Use the `policy-research-agent` to perform in-depth research
3️ Write a detailed report to `final_report.md`
4️ Optionally, ask the `policy-critique-agent` to critique your draft
5️ Revise if necessary, then output the final, comprehensive report

When writing the final report:
- Use Markdown with clear sections (## for each)
- Include citations in [Title](URL) format
- Add a ### Sources section at the end
- Write in professional, neutral tone suitable for policy briefings
"""

from langchain_ollama import ChatOllama
model = ChatOllama(model="qwen2.5:14b",temperature=0)
agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=policy_research_instructions,
    subagents=[research_sub_agent, critique_sub_agent],
)

query = "What are the latest updates on the EU AI Act and its global impact?"
result = agent.invoke({"messages": [{"role": "user", "content": query}]})

# for msg in result["messages"]:
#     print(msg)

from typing import Any

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

parsed = parse_messages(result)

print(parsed)