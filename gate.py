import time
import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("API Key not found! Add GOOGLE_API_KEY to .env")

GATEWAY_CONFIG = {
    "llm_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
    "api_key": API_KEY
}

MESSAGE_LOGS = []

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.post("/configure")
async def configure_gateway(config: dict):
    GATEWAY_CONFIG["llm_url"] = config.get("llm_url")
    GATEWAY_CONFIG["api_key"] = config.get("api_key")
    return {"status": "configured", "config": GATEWAY_CONFIG}


async def call_llm_and_broadcast(body: dict):
    """
    Call LLM async and broadcast result to all dashboard clients.
    """
    start = time.time()
    url = GATEWAY_CONFIG["llm_url"]
    if GATEWAY_CONFIG["api_key"]:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}key={GATEWAY_CONFIG['api_key']}"

    # Extract prompt from Gemini Flash payload
    try:
        prompt = body["contents"][0]["parts"][0]["text"]
    except (KeyError, IndexError):
        prompt = str(body)  # fallback

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(url, json=body)
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

    try:
        model_output = data["content"][0]["text"]
    except (KeyError, IndexError):
        model_output = str(data)

    meta = {
        "status_code": resp.status_code,
        "latency_ms": latency,
        "request_bytes": len(str(body)),
        "response_bytes": len(str(data)),
        "token_usage": data.get("tokenUsage", {}),
        "cost_usd": None
    }

    log_entry = {"prompt": prompt, "response": model_output, "meta": meta}
    MESSAGE_LOGS.append(log_entry)

    # Broadcast to all dashboard clients
    await manager.broadcast(log_entry)


@app.post("/chat")
async def gateway_chat(request: Request, background_tasks: BackgroundTasks):
    """
    Receives Gemini Flash JSON { "contents": [{"parts": [{"text": "..."}]}] } from external app
    and processes LLM asynchronously.
    """
    if not GATEWAY_CONFIG["llm_url"]:
        return JSONResponse({"error": "Gateway not configured"}, status_code=400)

    body = await request.json()
    background_tasks.add_task(call_llm_and_broadcast, body)
    return {"status": "processing", "prompt": body.get("contents", [{}])[0].get("parts", [{}])[0].get("text")}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Dashboard WebSocket connection.
    """
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
