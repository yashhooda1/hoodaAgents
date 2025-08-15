import os
from dotenv import load_dotenv

# Load local .env (Streamlit Cloud uses Secrets automatically)
load_dotenv()

from langgraph.prebuilt import create_react_agent
from langchain import hub
from langchain_core.messages import SystemMessage, HumanMessage

# Tools
from langchain_community.tools.tavily_search import TavilySearchResults
from tools.custom_tools import calculator_tool
from memory.memory_config import get_memory

# --------- Config ---------
DEFAULT_SYSTEM_PROMPT = (
    "You are hoodaAgents — fast, helpful, and concise. "
    "Cite facts only when you used search tools. Ask for missing info only if critical."
)

def build_tools():
    tavily_key = os.getenv("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")  # set this in .env or Streamlit Secrets
    tavily = TavilySearchResults(max_results=3, tavily_api_key=tavily_key) if tavily_key else TavilySearchResults(max_results=3)
    return [tavily, calculator_tool]

def build_llm():
    """
    Choose backend via env:
      - LLM_BACKEND=openai (default; works on Streamlit Cloud)
      - LLM_BACKEND=ollama (local dev; requires langchain-ollama and a running daemon)
    """
    backend = os.getenv("LLM_BACKEND", "openai").lower()

    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "LLM_BACKEND=ollama but 'langchain-ollama' is not installed. "
                "Install it or set LLM_BACKEND=openai."
            ) from e
        model = os.getenv("OLLAMA_MODEL", "mistral")   # or llama3.1, etc.
        temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
        return ChatOllama(model=model, temperature=temperature)

    # OpenAI (default)
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI backend selected but 'langchain-openai' is missing. "
            "Add it to requirements.txt."
        ) from e

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "Missing OPENAI_API_KEY. Set it in .env (local) or Streamlit Secrets (cloud)."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    return ChatOpenAI(model=model, temperature=temperature)

def load_prompt():
    """Try to load a prompt from the LangChain Hub; fall back to a local system prompt."""
    try:
        # Your original: "wfh/react-agent-executor"
        # Keep it, but if hub auth/network fails, fall back.
        return hub.pull("wfh/react-agent-executor")
    except Exception:
        # Fallback: simple SystemMessage prompt
        return SystemMessage(content=DEFAULT_SYSTEM_PROMPT)

def _build_agent():
    llm = build_llm()
    tools = build_tools()
    memory = get_memory()

    # Bind tools (enables tool-calls on OpenAI and Ollama backends)
    llm_with_tools = llm.bind_tools(tools)

    prompt = load_prompt()  # hub prompt or fallback SystemMessage
    agent = create_react_agent(llm_with_tools, tools, messages_modifier=prompt)
    agent.memory = memory
    return agent

# ---------- Public API ----------
def run_agent(user_text: str) -> str:
    """Single-turn entry point used by Streamlit main.py"""
    agent = _build_agent()
    response = agent.invoke({"messages": [("user", user_text)]})
    # response['messages'] is a list of messages; usually last is the assistant
    # Extract the last assistant message content
    messages = response.get("messages", [])
    if not messages:
        return "No response generated."
    # Find last non-empty content
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or (isinstance(msg, tuple) and msg[-1]) or ""
        if content:
            return content
    return "No response generated."

# ---------- Optional: CLI loop for local testing ----------
if __name__ == "__main__":
    print("🤖 hoodaAgents CLI — Ctrl+C to exit.")
    agent = _build_agent()
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            result = agent.invoke({"messages": [("user", user_input)]})
            for m in result.get("messages", []):
                if getattr(m, "type", None) == "ai" or getattr(m, "role", None) in ("assistant", None):
                    print("Agent:", getattr(m, "content", ""))
    except KeyboardInterrupt:
        print("\nGoodbye!")

