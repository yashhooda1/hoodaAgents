# 🚀 Deployment Guide for hoodaAgents

This guide shows you how to deploy hoodaAgents locally or online.

---

## ✅ Local Setup

### 1. Install Dependencies

```bash
pip install -r config/requirements.txt streamlit
```

### 2. Install Ollama & Pull Model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull hoodarunner/hoodaAgents
```

### 3. Run Locally in Terminal

```bash
python main.py
```

### 4. Run Web UI with Streamlit

```bash
streamlit run web/app.py
```

---

## 🌐 Optional: Deploy on Hugging Face Spaces

- Create a `Space` with Python + Streamlit runtime
- Upload your files
- Include a `requirements.txt` with `ollama`, `streamlit`, etc.
- Set up a `Modelfile` and configure Ollama on the backend

---

## ✈️ Optional: Deploy on Fly.io or Render.com

- Containerize the app with `Dockerfile` (not included yet)
- Use Fly.io’s CLI or Render's web UI to deploy
- You may need to host Ollama separately or use CPU-based models

---

Built with ❤️ by [Yash Hooda](https://github.com/yashhooda1)
