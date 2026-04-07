import asyncio
import json
import httpx
import subprocess
import time
from typing import Optional
from microgrid_env.models import MicrogridAction, MicrogridObservation, MicrogridResult


class MicrogridEnv:
    """
    HTTP client for the Microgrid OpenEnv environment.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        self._container_id: Optional[str] = None

    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 7860) -> "MicrogridEnv":
        """Pull and run from a local Docker image."""
        print(f"[INFO] Starting Docker container from {image_name}...")
        result = subprocess.run(
            ["docker", "run", "-d", "-p", f"{port}:7860", image_name],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker run failed: {result.stderr}")

        container_id = result.stdout.strip()
        instance = cls(base_url=f"http://localhost:{port}")
        instance._container_id = container_id

        # Wait for server to be ready
        for _ in range(30):
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"http://localhost:{port}/health", timeout=2)
                    if r.status_code == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(1)

        return instance

    async def reset(self, task: str = "load_balance") -> MicrogridResult:
        r = await self._client.post("/reset", json={"task": task})
        r.raise_for_status()
        data = r.json()
        obs = MicrogridObservation(**data)
        return MicrogridResult(
            observation=obs,
            reward=0.0,
            done=False,
            info={}
        )

    async def step(self, action: MicrogridAction) -> MicrogridResult:
        r = await self._client.post("/step", json=action.model_dump())
        r.raise_for_status()
        data = r.json()
        return MicrogridResult(
            observation=MicrogridObservation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data.get("info", {})
        )

    async def get_state(self) -> dict:
        r = await self._client.get("/state")
        r.raise_for_status()
        return r.json()

    async def close(self):
        await self._client.aclose()
        if self._container_id:
            subprocess.run(["docker", "stop", self._container_id], capture_output=True)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()