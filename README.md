---
title: Microgrid Env
emoji: 💡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI Custodire Resilience Engine for Next-Gen Microgrids

## Why This Matters

Power grids are failing. Climate change brings unpredictable solar generation. EV adoption spikes demand overnight. A single fault can cascade into blackouts affecting millions. Today, human operators make split-second decisions about battery dispatch, load shedding, and fault isolation under pressure, with incomplete information.

AI Custodire trains RL agents to do this automatically. An agent that masters microgrid management could prevent blackouts, reduce energy waste, and accelerate the clean energy transition. This is not a toy problem - utilities like ERCOT, NTPC, and Adani Green face these exact challenges daily.

## Environment Description

A simulated microgrid with solar generation, battery storage, variable load demand, and circuit switching. The agent acts as an autonomous grid operator, balancing power supply and demand while responding to faults and optimizing operational cost.

## Action Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| battery_dispatch | float | -10 to +10 MW | Positive = discharge (supply power), Negative = charge (store power) |
| load_shed | float | 0.0 to 1.0 | Fraction of load to curtail (0 = none, 1 = cut all) |
| switch_cmd | int | 0-3 | 0=no change, 1=open seg1, 2=open seg2, 3=restore all |

## Observation Space

| Field | Type | Normal Range | Description |
|-------|------|-------------|-------------|
| voltage_pu | float | 0.95-1.05 | Grid voltage in per-unit |
| frequency_hz | float | 49.5-50.5 | Grid frequency in Hz |
| load_mw | float | 2-15 MW | Total demand |
| solar_mw | float | 0.5-10 MW | Solar generation |
| battery_soc | float | 0-1 | Battery state of charge |
| net_balance_mw | float | - | Generation minus demand |
| fault_active | bool | - | Active fault present |
| fault_segment | int | 0-2 | Faulted segment (0=none) |
| seg1_energized | bool | - | Segment 1 powered |
| seg2_energized | bool | - | Segment 2 powered |

## Tasks

| Task | Difficulty | Steps | Objective |
|------|-----------|-------|-----------|
| load_balance | Easy | 20 | Keep net_balance_mw near zero across variable solar and load |
| fault_recovery | Medium | 30 | Detect fault at step 6, isolate faulted segment, restore grid |
| optimal_dispatch | Hard | 40 | Minimize battery cycling cost while maintaining voltage and frequency stability |

## Reward Design

Rewards are dense every step - agents get continuous feedback, not just end-of-episode:

- load_balance: 0-1 per step based on balance quality. Penalizes load shedding and active faults
- fault_recovery: -0.4 per step with unhandled fault. +0.4 bonus for correct isolation. 0-0.6 for post-recovery stability
- optimal_dispatch: Stability score minus operational cost. Clamped to [0.0, 1.0]

## Setup

pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

## Docker

docker build -t microgrid-env .
docker run -d -p 7860:7860 microgrid-env

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check |
| /reset | POST | Reset with task name |
| /step | POST | Execute action |
| /state | GET | Current state |
| /tasks | GET | List tasks with graders |
| /grader | POST | Score a task episode |
| /metadata | GET | Environment metadata |
| /schema | GET | Action and observation schemas |

## Baseline Scores (Qwen2.5-72B-Instruct)

| Task | Score | Success |
|------|-------|---------|
| load_balance | 0.920 | true |
| fault_recovery | 0.351 | true |
| optimal_dispatch | 0.776 | true |
