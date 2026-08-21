# Deployment and publishing

hoodaAgents has two independently deployable pieces:

1. The Ollama model produced from `Modelfile`
2. The optional Python runtime that adds memory, tools, evaluations, CLI, and UI

## Local installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r config/requirements.txt
cp .env.example .env

ollama pull qwen3.5:9b
ollama create hoodaAgents -f Modelfile
OLLAMA_MODEL=hoodaAgents python main.py
```

Ollama must remain reachable at `OLLAMA_BASE_URL`. The default is
`http://localhost:11434`.

## Validate before publishing

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
python evals/run_evals.py --model hoodaAgents
```

Do not publish a model that fails the configured live-evaluation threshold.

## Publish on ollama.com

```bash
ollama signin
ollama cp hoodaAgents hoodarunner/hoodaAgents
ollama push hoodarunner/hoodaAgents
```

Confirm from a clean model name:

```bash
ollama rm hoodarunner/hoodaAgents
ollama run hoodarunner/hoodaAgents
```

## Streamlit UI

```bash
OLLAMA_MODEL=hoodaAgents streamlit run web/app.py
```

The UI server and Ollama may run on different hosts by setting
`OLLAMA_BASE_URL`. Do not expose an unauthenticated Ollama daemon directly to
the public internet.

## Secrets and state

- Store configuration in an ignored `.env`, never in Git.
- SQLite memory defaults to `~/.hoodaagents/memory.db`.
- Set `HOODA_MEMORY=off` for stateless or shared deployments.
- `TAVILY_API_KEY` is optional; without it, web search is not registered.
