from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain import hub
from tools.custom_tools import calculator_tool
from memory.memory_config import get_memory

import os
from dotenv import load_dotenv
load_dotenv()
tavily_api_key = os.getenv("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")
tools = [TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key), calculator_tool]

def run_agent():
    llm = ChatOllama(model="mistral")
    tools = [TavilySearchResults(max_results=3), calculator_tool]
    memory = get_memory()

    llm_with_tools = llm.bind_tools(tools)
    prompt = hub.pull("wfh/react-agent-executor")

    agent_executor = create_react_agent(llm_with_tools, tools, messages_modifier=prompt)
    agent_executor.memory = memory

    print("🤖 hoodaAgent is ready. Type your question or Ctrl+C to exit.")
    while True:
        try:
            user_input = input("You: ")
            response = agent_executor.invoke({"messages": [("user", user_input)]})
            for message in response['messages']:
                print("Agent:", message.content)
        except KeyboardInterrupt:
            print("Exiting. Goodbye!")
            break
