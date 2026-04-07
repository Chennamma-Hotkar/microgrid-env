from pydantic import BaseModel, Field
from typing import Optional


class MicrogridAction(BaseModel):
    """
    Action space for the Microgrid environment.
    battery_dispatch: MW output from battery (-10=charge, +10=discharge)
    load_shed: fraction of load to shed (0.0 = no shedding, 1.0 = full shed)
    switch_cmd: 0=no change, 1=open segment1, 2=open segment2, 3=restore all
    """
    battery_dispatch: float = Field(default=0.0, ge=-10.0, le=10.0)
    load_shed: float = Field(default=0.0, ge=0.0, le=1.0)
    switch_cmd: int = Field(default=0, ge=0, le=3)


class MicrogridObservation(BaseModel):
    """
    Observation space for the Microgrid environment.
    """
    voltage_pu: float          # Grid voltage in per-unit (normal: 0.95–1.05)
    frequency_hz: float        # Grid frequency in Hz (normal: 49.5–50.5)
    load_mw: float             # Total demand in MW
    solar_mw: float            # Solar generation in MW
    battery_soc: float         # Battery state of charge (0.0–1.0)
    battery_mw: float          # Current battery output in MW
    net_balance_mw: float      # Generation minus load (positive = surplus)
    fault_active: bool         # True if an active fault exists
    fault_segment: int         # 0=none, 1=seg1, 2=seg2
    seg1_energized: bool       # Is segment 1 powered?
    seg2_energized: bool       # Is segment 2 powered?
    step: int                  # Current step number
    max_steps: int             # Max steps this episode
    task: str                  # Current task name
    episode_reward: float      # Cumulative reward so far


class MicrogridResult(BaseModel):
    observation: MicrogridObservation
    reward: float
    done: bool
    info: dict = {}