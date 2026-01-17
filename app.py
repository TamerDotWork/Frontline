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

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------
# App Config
# ---------------------------
app.state.config = {
    "api_key": "",
    "provider": "",
    "base_url": "",
    "model": "",
    "input_rate": 0.0,
    "output_rate": 0.0,
}

# Logs and counters
MESSAGE_LOGS = []
PROJECT_COUNTERS = {}
LOG_DIR = Path("storage/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Helpers
# ---------------------------
def save_log_to_file(entry: dict):
    """Append log to a JSONL file (no DB, append-only)."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    file_path = LOG_DIR / f"{date}.jsonl"
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def extract_tokens(data: dict) -> tuple[int, int]:
    """Extract input and output token counts from any provider."""
    usage = data.get("usageMetadata") or data.get("usage") or {}

    # Gemini / Google
    in_tokens = usage.get("inputTokens") or usage.get("promptTokenCount") or 0
    out_tokens = usage.get("outputTokens") or usage.get("candidatesTokenCount") or 0

    # OpenAI
    if in_tokens == 0 and out_tokens == 0:
        in_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        out_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0

    return in_tokens, out_tokens


def update_project_counters(project_id: str, meta: dict):
    """Maintain running totals: messages + tokens + cost per session."""
    if project_id not in PROJECT_COUNTERS:
        PROJECT_COUNTERS[project_id] = {
            "total_messages": 0,
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_tokens": 0,
            "total_success": 0,
            "total_cost": 0.0,
            "last_message_cost": 0.0,
            "success_percent": 0.0,
        }

    counters = PROJECT_COUNTERS[project_id]

    usage = meta.get("token_usage", {})
    in_tokens = usage.get("input_tokens", 0)
    out_tokens = usage.get("output_tokens", 0)
    total_tokens = in_tokens + out_tokens

    # Calculate cost per message
    input_rate = float(app.state.config.get("input_rate", 0) or 0)
    output_rate = float(app.state.config.get("output_rate", 0) or 0)
    cost = (in_tokens / 1000 * input_rate) + (out_tokens / 1000 * output_rate)

    # Update counters
    counters["total_messages"] += 1
    counters["total_tokens_input"] += in_tokens
    counters["total_tokens_output"] += out_tokens
    counters["total_tokens"] += total_tokens
    counters["total_cost"] += cost
    counters["last_message_cost"] = round(cost, 6)

    if 200 <= meta.get("status_code", 0) < 300:
        counters["total_success"] += 1

    counters["success_percent"] = round(
        (counters["total_success"] / counters["total_messages"]) * 100, 2
    )

    return counters


# ---------------------------
# Web UI Routes
# ---------------------------
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
    model: str = Form(...),
    input_rate: str = Form(...),
    output_rate: str = Form(...),
):
    app.state.config.update({
        "api_key": api_key,
        "base_url": gateway_url,
        "provider": provider,
        "model": model,
        "input_rate": float(input_rate or 0),
        "output_rate": float(output_rate or 0)
    })
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


# ---------------------------
# Prompt Normalization
# ---------------------------
def normalize_prompt(body: dict) -> str:
    try:
        return body["contents"][0]["parts"][0]["text"]  # Gemini
    except Exception:
        pass
    try:
        return body["messages"][-1]["content"]  # OpenAI
    except Exception:
        pass
    if "prompt" in body:  # Ollama
        return body["prompt"]
    return str(body)


# ---------------------------
# Provider URL Resolver
# ---------------------------
def resolve_provider_url():
    provider = app.state.config.get("provider")
    base_url = app.state.config.get("base_url")
    model = app.state.config.get("model")

    if provider == "ollama":
        if base_url and not base_url.endswith("/api/chat"):
            base_url = base_url.rstrip("/") + "/api/chat"
        return base_url
    if provider in ["gemini", "gemma3", "google"]:
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    if provider == "openai":
        return "https://api.openai.com/v1/chat/completions"
    return base_url


# ---------------------------
# Payload Builder
# ---------------------------
def build_request_payload(prompt: str):
    provider = app.state.config.get("provider")
    model = app.state.config.get("model")
    if provider in ["gemini", "gemma3", "google"]:
        return {"contents": [{"parts": [{"text": prompt}]}]}
    if provider == "openai":
        return {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if provider == "ollama":
        return {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}]}
    return {"prompt": prompt}


# ---------------------------
# Provider Output Parsing
# ---------------------------
def parse_response_output(provider: str, data: dict):
    try:
        if provider in ["gemini", "gemma3", "google"]:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        if provider == "openai":
            return data["choices"][0]["message"]["content"]
        if provider == "ollama":
            return data["message"]["content"]
    except Exception:
        return str(data)
    return str(data)


# ---------------------------
# CHAT ENTRYPOINT
# ---------------------------
@app.post("/chat")
async def gateway_chat(request: Request, background_tasks: BackgroundTasks):
    if not resolve_provider_url():
        return JSONResponse({"error": "Provider not supported"}, status_code=400)

    body = await request.json()
    project_id = body.get("project_id", "default")
    body["_project_id"] = project_id

    background_tasks.add_task(call_llm_and_broadcast, body)

    prompt = normalize_prompt(body)
    return {"status": "processing", "project_id": project_id, "prompt": prompt}


# ---------------------------
# MAIN LLM CALL + METRICS (Unified & Robust)
# ---------------------------
async def call_llm_and_broadcast(body: dict):
    project_id = body.get("_project_id", "default")
    provider = app.state.config.get("provider")

    start = time.time()
    prompt = normalize_prompt(body)
    payload = build_request_payload(prompt)
    url = resolve_provider_url()
    headers = {}

    # Authorization
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
            latency = round((time.time() - start) * 1000, 2)
            totals = update_project_counters(project_id, {
                "status_code": 500,
                "latency_ms": latency,
                "provider": provider,
                "token_usage": {"input_tokens": 0, "output_tokens": 0}
            })
            log_entry = {
                "project_id": project_id,
                "prompt": prompt,
                "response": f"Error: {e}",
                "provider": provider,
                "status_code": 500,
                "latency_ms": latency,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
                "total_messages": totals["total_messages"],
                "total_tokens_input": totals["total_tokens_input"],
                "total_tokens_output": totals["total_tokens_output"],
                "total_tokens": totals["total_tokens"],
                "total_cost": totals["total_cost"],
                "last_message_cost": totals["last_message_cost"],
                "success_percent": totals["success_percent"],
                "raw_data": None,
                "success": False
            }
            MESSAGE_LOGS.append(log_entry)
            save_log_to_file(log_entry)
            await manager.broadcast(log_entry)
            return

    latency = round((time.time() - start) * 1000, 2)
    model_output = parse_response_output(provider, data)

    # Robust token extraction
    in_tokens, out_tokens = extract_tokens(data)

    totals = update_project_counters(project_id, {
        "status_code": resp.status_code,
        "latency_ms": latency,
        "provider": provider,
        "token_usage": {"input_tokens": in_tokens, "output_tokens": out_tokens}
    })

    # Unified log entry
    log_entry = {
        "project_id": project_id,
        "prompt": prompt,
        "response": model_output,
        "provider": provider,
        "status_code": resp.status_code,
        "latency_ms": latency,
        "token_usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
        "total_messages": totals["total_messages"],
        "total_tokens_input": totals["total_tokens_input"],
        "total_tokens_output": totals["total_tokens_output"],
        "total_tokens": totals["total_tokens"],
        "total_cost": totals["total_cost"],
        "last_message_cost": totals["last_message_cost"],
        "success_percent": totals["success_percent"],
        "raw_data": data,
        "success": 200 <= resp.status_code < 300
    }

    MESSAGE_LOGS.append(log_entry)
    save_log_to_file(log_entry)
    await manager.broadcast(log_entry)


# ---------------------------
# WebSocket Manager
# ---------------------------
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
        for conn in self.active_connections:
            await conn.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
