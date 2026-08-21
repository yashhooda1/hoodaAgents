"""Streamlit UI for the native local-first hoodaAgents runtime."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from hooda_agents import AgentError, Settings, build_agent
from hooda_agents.client import OllamaError


st.set_page_config(page_title="hoodaAgents", page_icon="🧠", layout="centered")
st.title("🧠 hoodaAgents")
st.caption("Native Ollama agent loop · local memory · allow-listed tools")

base_settings = Settings.from_env()
with st.sidebar:
    st.header("Runtime")
    model = st.text_input("Ollama model", value=base_settings.model)
    session_id = st.text_input("Memory session", value="streamlit")
    memory_enabled = st.toggle(
        "Local SQLite memory",
        value=base_settings.memory_enabled,
    )
    show_trace = st.toggle("Show tool trace", value=True)
    st.caption(f"Ollama: {base_settings.ollama_base_url}")


@st.cache_resource(show_spinner=False)
def get_agent(model_name: str, use_memory: bool):
    settings = replace(
        Settings.from_env(),
        model=model_name,
        memory_enabled=use_memory,
    )
    return build_agent(settings)


agent = get_agent(model, memory_enabled)

if st.sidebar.button("Clear this session"):
    agent.clear_memory(session_id)
    st.session_state.messages = []
    st.sidebar.success("Session memory cleared")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_events") and show_trace:
            with st.expander("Tool trace"):
                st.json(message["tool_events"])

prompt = st.chat_input("Ask hoodaAgents anything")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.status("Reasoning locally with Ollama…", expanded=False):
                result = agent.run(prompt, session_id=session_id)
            st.markdown(result.text)
            event_payload = [event.as_dict() for event in result.tool_events]
            if event_payload and show_trace:
                with st.expander("Tool trace"):
                    st.json(event_payload)
            if result.thinking and show_trace:
                with st.expander("Model thinking"):
                    st.write("\n\n".join(result.thinking))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.text,
                    "tool_events": event_payload,
                }
            )
        except (AgentError, OllamaError, ValueError) as exc:
            st.error(f"Agent failed safely: {exc}")
