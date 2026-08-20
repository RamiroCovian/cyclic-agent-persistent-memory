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

Variables relevantes:

| Variable | Descripción |
| --- | --- |
| `LLM_PROVIDER` | `gemini` \| `openai` \| `anthropic` |
| `LLM_MODEL` | Modelo a usar |
| `SQLITE_DB_PATH` | Ruta del checkpointer (default `./data/checkpoints.db`) |
| `RECURSION_LIMIT` | Tope de pasos del grafo (default `10`) |

### 2. Entorno virtual (venv)

```bash
python -m venv env

# Windows
env\Scripts\activate

# macOS / Linux
source env/bin/activate

pip install -r requirements.txt
```

### 3. Ejecutar tests unitarios

```bash
pytest -q
```

Los tests cubren tools/DB, config, LLM+`bind_tools`, topología del grafo, persistencia (`thread_id`/`recursion_limit`) y exportación de trazas. No requieren API keys reales.

### 4. Ejecutar la demo multi-paso

```bash
python -m agent.main
```

Eso:

1. Muestra la topología del `StateGraph` (`model` ↔ `tools` + `tools_condition`).
2. Verifica el vínculo `llm.bind_tools(...)`.
3. Corre una consulta que fuerza **≥2 tool calls** (resumen + último pedido del cliente 102).
4. Aplica `recursion_limit` desde el entorno.
5. Guarda la traza ReAct en:
   - `traces/react-multipaso.json`
   - `traces/react-multipaso.log`

Demo de memoria de sesión (mismo `thread_id`, dos turnos):

```bash
python -c "import asyncio; from agent.main import demostrar_memoria_sesion; asyncio.run(demostrar_memoria_sesion())"
```

## Estructura del proyecto

```text
.
├── agent/
│   ├── graph.py         # StateGraph + MessagesState + tools_condition
│   ├── llm.py           # LLM + bind_tools
│   ├── persistence.py   # SqliteSaver / AsyncSqliteSaver + thread_id
│   ├── trace.py         # Exportación de traza ReAct (.json / .log)
│   ├── main.py          # Demo multi-paso + recursion_limit
│   └── tools/           # Herramientas @tool (buscar_pedidos, etc.)
├── data/                # SQLite local (ignorado por git)
├── traces/              # Trazas de ejemplo incluidas en el repo
├── tests/               # Pruebas unitarias (pytest)
├── .env.example         # Plantilla sin secretos
└── requirements.txt     # Dependencias (incluye pytest)
```

## Traza de ejemplo

Ver `traces/react-multipaso.log` (también existe la versión JSON). Formato esperado:

```text
Usuario: "Del cliente 102, ¿cuántos pedidos tuvo... y el detalle del último?"
-> El agente decide usar la herramienta: buscar_pedidos(...)
-> La herramienta buscar_pedidos devuelve: {"pedidos": 3, "total": 14500.0}
-> El agente decide usar la herramienta: obtener_detalle_ultimo_pedido(...)
-> La herramienta obtener_detalle_ultimo_pedido devuelve: {...}
Respuesta: "..."
```

## Seguridad

- Las API keys viven solo en `.env` (listado en `.gitignore`).
- El repo no incluye claves ni archivos de credenciales.
- `data/*.db` está ignorado; las trazas en `traces/` no contienen secretos.
