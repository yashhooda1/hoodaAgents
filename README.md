# 🧠 hoodaAgents — Web AI Agent Framework Powered by OpenAI GPT-4

**hoodaAgents** is a next-gen, web-based AI assistant framework powered by **OpenAI GPT-4**, built using **LangChain**, **Tavily**, and **Streamlit**. This version of hoodaAgents runs entirely through a user-friendly browser interface and connects powerful LLM capabilities with real-time web tools and memory.

It serves as a production-ready template for building intelligent AI agents that can:

- 💬 Answer user queries with contextual memory
- 🔍 Perform live web research using Tavily
- 🔧 Be extended with custom tools (like a calculator)
- 🧠 Maintain dynamic multi-turn conversations
- 🌐 Run in your browser via Streamlit

---

## 🚀 Why hoodaAgents?

In an age of bloated, black-box AI apps, **hoodaAgents** gives you control and flexibility. Whether you're a developer, researcher, student, or builder — you can use this agent to power your personal assistant, test custom workflows, or extend it into your own AI product.

---

## 🛠️ Built With

- **OpenAI GPT-4** – LLM for reasoning and dialogue
- **LangChain** – Agent orchestration and tool routing
- **Tavily** – Fast real-time search for relevant info
- **Streamlit** – Clean browser-based UI
- **Python** – Modular and extensible backend

---

## 🧩 Project Use Cases

- Personal AI chatbot
- Web-based AI search agent
- Custom GPT-4 playground
- Prompt engineering sandbox
- Data-driven productivity tool

---

## 📁 Project Structure
hoodaAgents/
├── web/
│ └── app.py # Streamlit UI entry point
├── tools/
│ └── custom_tools.py # Calculator and other tools
├── memory/
│ └── memory_config.py # Memory setup using LangChain
├── config/
│ └── requirements.txt # Python dependencies
├── .env # API keys (keep this secret!)
└── README.md # Project documentation (you’re here!)


---

## 🧪 Getting Started

1. **Clone the repo**

```bash
git clone https://github.com/yashhooda1/hoodaAgents.git
cd hoodaAgents
```

2. **Install dependencies**

bash
Copy
Edit
pip install -r config/requirements.txt
Add your API keys

3. **Create a .env file in the root with:**

ini
Copy
Edit
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here

4. **Run the app**

bash
Copy
Edit
streamlit run web/app.py

🎥 Demo Video
▶️ Watch hoodaAgents in action (YouTube)

🛠️ Customize Your Agent
Modify web/app.py, custom_tools.py, or memory_config.py to:

Swap GPT-4 with other OpenAI models

Add new tools (e.g. weather, calendar, file parsing)

Integrate vector DBs or APIs

Change UI layout in Streamlit

✅ Live Demo on Website
Check it out on my personal website and GitHub profile.

Made with ❤️ by Yash Hooda
Let’s redefine what intelligent agents can do — your way.
