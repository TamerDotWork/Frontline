import time
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
    # Update global config
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

 

@app.post("/chat")
async def gateway_chat(request: Request, background_tasks: BackgroundTasks):
    if not resolve_provider_url():
        return JSONResponse({"error": "Provider not supported"}, status_code=400)

    body = await request.json()
    background_tasks.add_task(call_llm_and_broadcast, body)
    prompt = normalize_prompt(body)
    return {"status": "processing", "prompt": prompt}


def normalize_prompt(body: dict) -> str:
    # Gemini / Gemma
    try:
        return body["contents"][0]["parts"][0]["text"]
    except Exception:
        pass

    # OpenAI format
    try:
        return body["messages"][-1]["content"]
    except Exception:
        pass

    # Ollama fallback
    if "prompt" in body:
        return body["prompt"]

    return str(body)



def resolve_provider_url():
    provider = app.state.config.get("provider")
    base_url = app.state.config.get("base_url")
    model = app.state.config.get("model")

    if provider == "ollama":
        if base_url:
            # ensure the URL ends with /api/chat
            if not base_url.endswith("/api/chat"):
                base_url = base_url.rstrip("/") + "/api/chat"
            return base_url
        else:
            return None

    if provider in ["gemini", "gemma3","google"]:
        return (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent"
        )

    if provider == "openai":
        return "https://api.openai.com/v1/chat/completions"

    # fallback
    return base_url




def build_request_payload(prompt: str):
    provider = app.state.config["provider"]

    if provider in ["gemini", "gemma3","google"]:
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



def parse_response_output(provider: str, data: dict):
    if provider in ["gemini", "gemma3","google"]:
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



async def call_llm_and_broadcast(body: dict):
    provider = app.state.config["provider"]
    start = time.time()

    prompt = normalize_prompt(body)
    payload = build_request_payload(prompt)
    url = resolve_provider_url()
    headers = {}

    # Provider-specific auth
    if provider == "openai":
        headers["Authorization"] = f"Bearer {app.state.config['api_key']}"

    if provider in ["gemini", "gemma2", "google"]:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}key={app.state.config['api_key']}"

    async with httpx.AsyncClient(timeout=40) as client:
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log_entry = {
                "prompt": prompt,
                "response": f"Error: {str(e)}",
                "meta": {"status_code": 500, "latency_ms": 0}
            }
            MESSAGE_LOGS.append(log_entry)
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

    log_entry = {"prompt": prompt, "response": model_output, "meta": meta,"data": data}
    MESSAGE_LOGS.append(log_entry)
    await manager.broadcast(log_entry)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)





 