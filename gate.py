import time
import json
from pathlib import Path
from datetime import datetime

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.state.config = {
    "api_key": "",
    "provider": "",
    "base_url": "",
    "model": ""
}

MESSAGE_LOGS = []

# running counters per session/project
PROJECT_COUNTERS = {}

# optional file storage directory (no DB)
LOG_DIR = Path("storage/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def save_log_to_file(entry: dict):
    """Append log to a JSONL file (no DB, append-only)."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    file_path = LOG_DIR / f"{date}.jsonl"
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_project_counters(project_id: str, meta: dict):
    """Maintain running totals: messages + tokens per session."""

    if project_id not in PROJECT_COUNTERS:
            PROJECT_COUNTERS[project_id] = {
                "total_messages": 0,
                "total_tokens_input": 0,
                "total_tokens_output": 0,
                "total_tokens": 0
            }

    counters = PROJECT_COUNTERS[project_id]

    usage = meta.get("token_usage", {}) or {}

    # normalize token fields across providers
    in_tokens = (
        usage.get("promptTokens")
        or usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or 0
    )

    out_tokens = (
        usage.get("candidatesTokens")
        or usage.get("completion_tokens")
        or usage.get("output_tokens")
        or 0
    )

    total_tokens = in_tokens + out_tokens

    counters["total_messages"] += 1
    counters["total_tokens_input"] += in_tokens
    counters["total_tokens_output"] += out_tokens
    counters["total_tokens"] += total_tokens

    return counters


# ----------------------------------------------------
# Web UI Routes
# ----------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "config.html",
        {"request": request, "config": app.state.config}
    )


@app.post("/save-config")
async def save_config(
    api_key: str = Form(...),
    gateway_url: str = Form(...),
    provider: str = Form(...),
    model: str = Form(...)
):
    app.state.config["base_url"] = gateway_url
    app.state.config["provider"] = provider
    app.state.config["api_key"] = api_key
    app.state.config["model"] = model

    print("Updated Config:", app.state.config)
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "config": app.state.config}
    )


@app.get("/projects", response_class=HTMLResponse)
async def projects(request: Request):
    return templates.TemplateResponse(
        "projects.html",
        {"request": request, "config": app.state.config}
    )


# ----------------------------------------------------
# Prompt Normalization
# ----------------------------------------------------

def normalize_prompt(body: dict) -> str:
    # Gemini / Gemma
    try:
        return body["contents"][0]["parts"][0]["text"]
    except Exception:
        pass

    # OpenAI
    try:
        return body["messages"][-1]["content"]
    except Exception:
        pass

    # Ollama
    if "prompt" in body:
        return body["prompt"]

    return str(body)


# ----------------------------------------------------
# Provider URL Resolver
# ----------------------------------------------------

def resolve_provider_url():
    provider = app.state.config.get("provider")
    base_url = app.state.config.get("base_url")
    model = app.state.config.get("model")

    if provider == "ollama":
        if base_url:
            if not base_url.endswith("/api/chat"):
                base_url = base_url.rstrip("/") + "/api/chat"
            return base_url
        return None

    if provider in ["gemini", "gemma3", "google"]:
        return (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent"
        )

    if provider == "openai":
        return "https://api.openai.com/v1/chat/completions"

    return base_url


# ----------------------------------------------------
# Payload Builder
# ----------------------------------------------------

def build_request_payload(prompt: str):
    provider = app.state.config["provider"]

    if provider in ["gemini", "gemma3", "google"]:
        return {"contents": [{"parts": [{"text": prompt}]}]}

    if provider == "openai":
        return {
            "model": app.state.config["model"],
            "messages": [{"role": "user", "content": prompt}]
        }

    if provider == "ollama":
        return {
            "model": app.state.config["model"],
            "stream": False,
            "messages": [{"role": "user", "content": prompt}]
        }

    return {"prompt": prompt}


# ----------------------------------------------------
# Provider Output Parsing
# ----------------------------------------------------

def parse_response_output(provider: str, data: dict):
    if provider in ["gemini", "gemma3", "google"]:
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return str(data)

    if provider == "openai":
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return str(data)

    if provider == "ollama":
        try:
            return data["message"]["content"]
        except Exception:
            return str(data)

    return str(data)


# ----------------------------------------------------
# CHAT ENTRYPOINT
# ----------------------------------------------------

@app.post("/chat")
async def gateway_chat(request: Request, background_tasks: BackgroundTasks):
    if not resolve_provider_url():
        return JSONResponse({"error": "Provider not supported"}, status_code=400)

    body = await request.json()

    # project/session id provided by client
    project_id = body.get("project_id", "default")
    body["_project_id"] = project_id

    background_tasks.add_task(call_llm_and_broadcast, body)

    prompt = normalize_prompt(body)

    return {
        "status": "processing",
        "project_id": project_id,
        "prompt": prompt
    }


# ----------------------------------------------------
# MAIN LLM CALL + METRICS
# ----------------------------------------------------

async def call_llm_and_broadcast(body: dict):
    from fastapi import WebSocket

    project_id = body.get("_project_id", "default")
    provider = app.state.config["provider"]

    start = time.time()

    prompt = normalize_prompt(body)
    payload = build_request_payload(prompt)
    url = resolve_provider_url()
    headers = {}

    # auth
    if provider == "openai":
        headers["Authorization"] = f"Bearer {app.state.config['api_key']}"

    if provider in ["gemini", "gemma3", "google"]:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}key={app.state.config['api_key']}"

    async with httpx.AsyncClient(timeout=40) as client:
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            meta = {
                "status_code": 500,
                "latency_ms": 0,
                "provider": provider,
                "token_usage": {}
            }

            totals = update_project_counters(project_id, meta)
            meta["session_totals"] = totals
            meta["message_index"] = totals["total_messages"]

            log_entry = {
                "project_id": project_id,
                "prompt": prompt,
                "response": f"Error: {str(e)}",
                "meta": meta
            }

            MESSAGE_LOGS.append(log_entry)
            save_log_to_file(log_entry)

            await manager.broadcast(log_entry)
            return

    latency = round((time.time() - start) * 1000, 2)
    model_output = parse_response_output(provider, data)

    meta = {
        "status_code": resp.status_code,
        "latency_ms": latency,
        "provider": provider,
        "request_bytes": len(str(payload)),
        "response_bytes": len(str(data)),
        "token_usage": data.get("usage", data.get("tokenUsage", {}))
    }

    # 🔹 update running totals (MESSAGE + TOKENS)
    totals = update_project_counters(project_id, meta)

    # attach counters to meta
    meta["session_totals"] = totals
    meta["message_index"] = totals["total_messages"]

    log_entry = {
        "project_id": project_id,
        "prompt": prompt,
        "response": model_output,
        "meta": meta,
        "data": data
    }

    MESSAGE_LOGS.append(log_entry)
    save_log_to_file(log_entry)

    await manager.broadcast(log_entry)


# ----------------------------------------------------
# WebSocket
# ----------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
