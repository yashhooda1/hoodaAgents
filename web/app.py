import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from tools.custom_tools import calculator_tool
from langgraph.prebuilt import create_react_agent
from langchain import hub
from memory.memory_config import get_memory

st.set_page_config(page_title="hoodaAgents", layout="centered")

st.title("🧠 hoodaAgents Web UI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatOllama(model="mistral")
tools = [TavilySearchResults(max_results=3), calculator_tool]
memory = get_memory()

llm_with_tools = llm.bind_tools(tools)
prompt = hub.pull("wfh/react-agent-executor")
agent_executor = create_react_agent(llm_with_tools, tools, messages_modifier=prompt)
agent_executor.memory = memory

user_input = st.text_input("Ask me anything:", key="input")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    response = agent_executor.invoke({"messages": [("user", user_input)]})
    for message in response['messages']:
        st.session_state.chat_history.append(("agent", message.content))

for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role.title()}**: {msg}")
