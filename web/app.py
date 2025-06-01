import sys
import os
from dotenv import load_dotenv

# ✅ Set up project path and environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

# ✅ Get API keys
tavily_key = os.getenv("TAVILY_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if not tavily_key:
    raise RuntimeError("TAVILY_API_KEY not found. Please check your .env file.")
if not openai_key:
    raise RuntimeError("OPENAI_API_KEY not found. Please check your .env file.")

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from tools.custom_tools import calculator_tool
from langchain.agents import initialize_agent, AgentType
from memory.memory_config import get_memory

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
