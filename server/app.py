import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from server.environment import MicrogridEnvironment
from microgrid_env.models import MicrogridAction

app = FastAPI(
    title="Microgrid Resilience Environment",
    description="AI Custodire Resilience Engine for Next-Gen Microgrids — OpenEnv",
    version="1.0.0",
)

# One environment instance per server (stateful)
env = MicrogridEnvironment()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(body: dict = {}):
    task = body.get("task", "load_balance")
    obs = env.reset(task=task)
    return obs.model_dump()


@app.post("/step")
def step(body: dict):
    try:
        action = MicrogridAction(**body)
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    result = env.step(action)
    return result


@app.get("/state")
def state():
    return env.get_state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "load_balance",
                "difficulty": "easy",
                "description": "Keep power supply and demand balanced for 20 steps.",
                "max_steps": 20,
            },
            {
                "id": "fault_recovery",
                "difficulty": "medium",
                "description": "Detect and isolate a segment fault, restore grid stability within 30 steps.",
                "max_steps": 30,
            },
            {
                "id": "optimal_dispatch",
                "difficulty": "hard",
                "description": "Minimize operational cost while maintaining voltage and frequency stability under variable solar and load for 40 steps.",
                "max_steps": 40,
            },
        ]
    }


# ─── WebSocket session (OpenEnv standard) ────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_env = MicrogridEnvironment()
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "reset":
                task = msg.get("data", {}).get("task", "load_balance")
                obs = session_env.reset(task=task)
                await websocket.send_text(json.dumps(obs.model_dump()))

            elif msg_type == "step":
                action = MicrogridAction(**msg.get("data", {}))
                result = session_env.step(action)
                await websocket.send_text(json.dumps(result))

            elif msg_type == "state":
                await websocket.send_text(json.dumps(session_env.get_state()))

            else:
                await websocket.send_text(json.dumps({"error": f"Unknown type: {msg_type}"}))

    except WebSocketDisconnect:
        pass


def run():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)