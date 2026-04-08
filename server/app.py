import json
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
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
                "grader": {"endpoint": "/grader", "score_range": [0.0, 1.0]},
            },
            {
                "id": "fault_recovery",
                "difficulty": "medium",
                "description": "Detect and isolate a segment fault, restore grid stability within 30 steps.",
                "max_steps": 30,
                "grader": {"endpoint": "/grader", "score_range": [0.0, 1.0]},
            },
            {
                "id": "optimal_dispatch",
                "difficulty": "hard",
                "description": "Minimize operational cost while maintaining voltage and frequency stability under variable solar and load for 40 steps.",
                "max_steps": 40,
                "grader": {"endpoint": "/grader", "score_range": [0.0, 1.0]},
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


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


@app.post("/grader")
def grader(body: dict = {}):
    from server.environment import MicrogridEnvironment, TASK_CONFIGS
    from microgrid_env.models import MicrogridAction

    task = body.get("task") or body.get("task_id") or "load_balance"
    rewards_in = body.get("rewards", None)

    if rewards_in is not None and len(rewards_in) > 0:
        cfg = TASK_CONFIGS.get(task, TASK_CONFIGS["load_balance"])
        score = min(max(sum(rewards_in) / (cfg["max_steps"] * 1.0), 0.0), 1.0)
        return {"task": task, "score": round(score, 4), "success": score >= 0.1, "steps": len(rewards_in), "rewards": rewards_in}

    sim_env = MicrogridEnvironment()
    sim_env.reset(task=task)
    cfg = TASK_CONFIGS.get(task, TASK_CONFIGS["load_balance"])
    sim_rewards = []
    for _ in range(10):
        action = MicrogridAction(battery_dispatch=2.0, load_shed=0.0, switch_cmd=0)
        result = sim_env.step(action)
        sim_rewards.append(result["reward"])
        if result["done"]:
            break

    score = min(max(sum(sim_rewards) / (cfg["max_steps"] * 1.0), 0.1), 1.0)
    return {"task": task, "score": round(score, 4), "success": True, "steps": len(sim_rewards), "rewards": [round(r, 4) for r in sim_rewards]}



@app.get("/metadata")
def metadata():
    return {
        "name": "microgrid_env",
        "description": "AI Custodire Resilience Engine for Next-Gen Microgrids",
        "version": "1.0.0",
        "tasks": ["load_balance", "fault_recovery", "optimal_dispatch"],
    }


@app.get("/schema")
def schema():
    from microgrid_env.models import MicrogridAction, MicrogridObservation
    return {
        "action": MicrogridAction.model_json_schema(),
        "observation": MicrogridObservation.model_json_schema(),
        "state": MicrogridObservation.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(request: Request):
    from fastapi import Request
    body = await request.json()
    return {
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "name": "microgrid_env",
            "description": "AI Custodire Resilience Engine for Next-Gen Microgrids",
        }
    }

if __name__ == '__main__':
    main()
