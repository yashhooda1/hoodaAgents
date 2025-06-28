import sys
import os
import streamlit as st

# ✅ Set up project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from tools.custom_tools import calculator_tool
from langchain.agents import initialize_agent, AgentType
from memory.memory_config import get_memory

# ✅ Use Streamlit Secrets for API keys
tavily_key = st.secrets["TAVILY_API_KEY"]
openai_key = st.secrets["OPENAI_API_KEY"]

# ✅ Streamlit setup
st.set_page_config(page_title="hoodaAgents", layout="centered")
st.title("🧠 hoodaAgents Web UI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ GPT-4 LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=openai_key
)

# ✅ Tools and memory
tools = [
    TavilySearchResults(max_results=3, tavily_api_key=tavily_key),
    calculator_tool
]
memory = get_memory()

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# ✅ User Input
user_input = st.text_input("Ask me anything:", key="input")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    # 👋 Friendly fallback
    casual_keywords = ["hi", "hello", "hey", "how are you", "what's up", "good morning", "good evening"]
    if any(greet in user_input.lower() for greet in casual_keywords):
        reply = "Hey! I'm doing great — ready to help you with anything. Ask me a question!"
    else:
        try:
            response = agent_executor.invoke({"input": user_input})
            reply = response["output"]
        except Exception as e:
            reply = f"⚠️ Agent failed to respond: {str(e)}"

    st.session_state.chat_history.append(("agent", reply))

# ✅ Display messages
for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role.title()}**: {msg}")
