"""
Baseline Inference Script — AI Custodire Resilience Engine for Next-Gen Microgrids
Follows OpenEnv stdout format strictly: [START] [STEP] [END]
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from microgrid_env import MicrogridEnv, MicrogridAction

# ─── Environment Variables ────────────────────────────────────────────────────
IMAGE_NAME    = os.getenv("LOCAL_IMAGE_NAME")         # optional: for docker
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ─── Task config ─────────────────────────────────────────────────────────────
TASKS = ["load_balance", "fault_recovery", "optimal_dispatch"]
BENCHMARK = "microgrid_env"
MAX_STEPS = 8
TEMPERATURE = 0.3
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.3

SYSTEM_PROMPT = textwrap.dedent("""
    You control a microgrid. Each turn you output a JSON action with exactly these keys:
    {"battery_dispatch": <float -10 to 10>, "load_shed": <float 0.0 to 1.0>, "switch_cmd": <int 0-3>}

    battery_dispatch: positive=discharge (supply power), negative=charge (store power)
    load_shed: fraction of load to cut (0=none, 1=cut everything) — use only when needed
    switch_cmd: 0=no change, 1=open segment1, 2=open segment2, 3=restore all

    Goals:
    - load_balance: keep net_balance_mw near 0 (supply = demand)
    - fault_recovery: if fault_active=true, use switch_cmd=1 to isolate, then switch_cmd=3 to restore
    - optimal_dispatch: maximize stability (voltage ~1.0, frequency ~50Hz), minimize battery cycling

    Reply with ONLY the JSON object, no explanation.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def parse_action(text: str) -> MicrogridAction:
    import json, re
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return MicrogridAction(
                battery_dispatch=float(data.get("battery_dispatch", 0.0)),
                load_shed=float(data.get("load_shed", 0.0)),
                switch_cmd=int(data.get("switch_cmd", 0)),
            )
    except Exception:
        pass
    return MicrogridAction(battery_dispatch=0.0, load_shed=0.0, switch_cmd=0)


def get_action(client: OpenAI, obs_dict: dict, step: int, history: List[str]) -> MicrogridAction:
    history_text = "\n".join(history[-3:]) if history else "None"
    user_prompt = f"Step {step}\nObservation: {obs_dict}\nPrevious steps:\n{history_text}\n\nOutput your action JSON:"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text)
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return MicrogridAction(battery_dispatch=0.0, load_shed=0.0, switch_cmd=0)


async def run_task(task_name: str, client: OpenAI) -> None:
    if IMAGE_NAME:
        env = await MicrogridEnv.from_docker_image(IMAGE_NAME)
    else:
        env = MicrogridEnv(base_url=os.getenv("ENV_URL", "http://localhost:7860"))

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_action(client, obs.model_dump(), step, history)
            action_str = f"battery={action.battery_dispatch:.1f},shed={action.load_shed:.2f},sw={action.switch_cmd}"

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"step={step} reward={reward:+.2f} balance={obs.net_balance_mw:.2f} fault={obs.fault_active}")

            if done:
                break

        max_possible = MAX_STEPS * 1.0
        score = min(max(sum(rewards) / max_possible, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task = os.getenv("MICROGRID_TASK", "load_balance")
    await run_task(task, client)


if __name__ == "__main__":
    asyncio.run(main())