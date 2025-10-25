from langchain.schema import HumanMessage, AIMessage
from agents.simple_agent import create_agent  # reuse your existing logic

_agent = create_agent()

def run_chat(query: str, history=None):
    messages = []
    if history:
        for role, msg in history:
            if role == "user":
                messages.append(HumanMessage(content=msg))
            else:
                messages.append(AIMessage(content=msg))
    messages.append(HumanMessage(content=query))
    result = _agent.invoke(messages)
    return result.content
