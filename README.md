# 🧠 hoodaAgents — Local AI Agent Framework Powered by Ollama

**hoodaAgents** is a local-first AI assistant framework designed to run customizable AI agents entirely on your machine using **Ollama** and **LangGraph**. It integrates powerful open-source LLMs (like Mistral) with intelligent tool usage, allowing your agents to interact with the world through web search, API calls, and more.

This project serves as the core engine for building modular, privacy-preserving, and efficient AI agents that can:

- 💬 Answer questions with contextual memory
- 🔍 Perform live web research (via Tavily)
- 🔧 Be extended with custom tools and workflows
- 🖥️ Run entirely offline (except search) with local models

---

## 🚀 Why hoodaAgents?

In a world flooded with cloud-dependent AI, **hoodaAgents** puts the power of AI back into your hands — literally. It lets developers, researchers, and curious builders spin up fully offline-capable AI agents that can perform intelligent tasks with zero reliance on OpenAI or big tech APIs.

---

## 🛠️ Built With

- **Ollama** – Run LLMs like Mistral locally
- **LangGraph** – Orchestrate agent flows with graphs
- **LangChain** – Connect LLMs with tools and memory
- **Tavily** – Real-time search tool for agents
- **Python** – Clean and modular codebase

---

## 🧩 Project Use Cases

- Personal AI assistants
- Local research tools
- Prompt engineering testing
- Privacy-preserving alternatives to ChatGPT
- Customizable AI workflows and automation

---

## 📁 Project Structure

```
hoodaAgents/
├── main.py                     # Entry point for the agent
├── agents/
│   └── simple_agent.py         # Agent setup using Ollama + LangChain
├── tools/                      # (Empty for now — add custom tools here)
├── config/
│   └── requirements.txt        # All dependencies
└── README.md                   # Full project documentation and usage
```

---

## 🧪 Getting Started

1. **Install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. **Pull the model**
```bash
ollama pull mistral
```

3. **Set up your Python environment**
```bash
pip install -r config/requirements.txt
```

4. **Run the agent**
```bash
python main.py
```

---

## 🛠️ Customize Your Agent

You can modify `simple_agent.py` to:
- Add or change tools
- Swap LLMs (e.g. LLaMA, Phi, Gemma)
- Build multi-agent systems
- Add memory, vector DB, or APIs

---

Made with ❤️ by Yash Hooda
