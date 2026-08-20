# cyclic-agent-persistent-memory

Agente autónomo con razonamiento cíclico ReAct, herramientas dinámicas, memoria persistente mediante LangGraph y SQLite, y ejecución asíncrona en Python 3.12+.

## Requisitos

- Python 3.12 o superior

## Configuración del entorno

### 1. Variables de entorno

```bash
cp .env.example .env
```

Editá `.env` y cargá tu API key (`GOOGLE_API_KEY`, `OPENAI_API_KEY` o `ANTHROPIC_API_KEY` según `LLM_PROVIDER`). **No subas el archivo `.env` al repositorio.**

### 2. Entorno virtual (venv)

```bash
python -m venv env

# Windows
env\Scripts\activate

# macOS / Linux
source env/bin/activate

pip install -r requirements.txt
python -m agent.main
```

`python -m agent.main` muestra la topología, el vínculo LLM y una **demo de memoria de sesión**: dos turnos con el mismo `thread_id` (el segundo pregunta "¿y el último?" y reutiliza el contexto del cliente 102). Los checkpoints se guardan en `SQLITE_DB_PATH` (por defecto `./data/checkpoints.db`).

## Estructura del proyecto

```text
.
├── agent/              # Código del agente (grafo, tools, checkpointer)
│   ├── graph.py        # StateGraph + tools_condition
│   ├── llm.py          # LLM + bind_tools
│   ├── persistence.py  # SqliteSaver / AsyncSqliteSaver + thread_id
│   └── tools/          # Herramientas @tool
├── data/               # SQLite local (ignorado por git)
├── traces/             # Trazas ReAct de ejemplo (.json / log)
├── .env.example        # Plantilla de variables (sin secretos)
└── requirements.txt    # Dependencias para pip/venv
```

## Seguridad

- Las API keys viven solo en `.env` (listado en `.gitignore`).
- El repo no incluye claves ni archivos de credenciales.
