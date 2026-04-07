# AI Custodire Resilience Engine for Next-Gen Microgrids

An OpenEnv RL environment simulating real-world microgrid power management.
An AI agent controls battery dispatch, load shedding, and circuit switching
to maintain grid stability, recover from faults, and optimize energy dispatch.

## Environment Description

**Real-world task**: Microgrid operators must continuously balance power supply
and demand, respond to faults, and minimize operational costs under uncertainty.

## Action Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| battery_dispatch | float | -10 to +10 MW | Positive = discharge, Negative = charge |
| load_shed | float | 0.0 to 1.0 | Fraction of load to curtail |
| switch_cmd | int | 0–3 | 0=none, 1=open seg1, 2=open seg2, 3=restore |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| voltage_pu | float | Grid voltage (normal: 0.95–1.05 pu) |
| frequency_hz | float | Grid frequency (normal: 49.5–50.5 Hz) |
| load_mw | float | Total demand in MW |
| solar_mw | float | Solar generation in MW |
| battery_soc | float | Battery state of charge (0–1) |
| net_balance_mw | float | Generation minus demand |
| fault_active | bool | Active fault present |
| fault_segment | int | Faulted segment (0=none) |
| seg1_energized | bool | Segment 1 powered |
| seg2_energized | bool | Segment 2 powered |

## Tasks

| Task | Difficulty | Steps | Description |
|------|-----------|-------|-------------|
| load_balance | Easy | 20 | Keep net_balance_mw near 0 |
| fault_recovery | Medium | 30 | Detect fault at step 6, isolate and restore |
| optimal_dispatch | Hard | 40 | Minimize cost under variable solar/load |

## Reward

- **load_balance**: 0–1 per step based on balance; penalties for shedding/faults
- **fault_recovery**: -0.4 per step with active fault; bonus for isolation; 0–1 for stability
- **optimal_dispatch**: stability score minus operational cost; range [-1, 1]

All scores normalized to [0.0, 1.0]

## Setup

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

## Docker

```bash
docker build -t microgrid-env -f server/Dockerfile .
docker run -d -p 7860:7860 microgrid-env
```

## Baseline Scores (approximate)

| Task | Random Agent | Baseline LLM |
|------|-------------|--------------|
| load_balance | 0.35 | 0.65 |
| fault_recovery | 0.10 | 0.45 |
| optimal_dispatch | 0.20 | 0.50 |