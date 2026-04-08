import random
import math
from microgrid_env.models import MicrogridAction, MicrogridObservation


TASK_CONFIGS = {
    "load_balance": {
        "max_steps": 20,
        "fault_at_step": None,
        "solar_variability": 0.05,
        "load_variability": 0.1,
    },
    "fault_recovery": {
        "max_steps": 30,
        "fault_at_step": 6,
        "solar_variability": 0.03,
        "load_variability": 0.05,
    },
    "optimal_dispatch": {
        "max_steps": 40,
        "fault_at_step": None,
        "solar_variability": 0.25,
        "load_variability": 0.3,
    },
}


class MicrogridEnvironment:
    def __init__(self):
        self._reset_internal()

    def _reset_internal(self):
        self.task = "load_balance"
        self.step_count = 0
        self.max_steps = 20
        self.fault_at_step = None
        self.solar_variability = 0.05
        self.load_variability = 0.1
        self.fault_active = False
        self.fault_segment = 0
        self.seg1_energized = True
        self.seg2_energized = True
        self.voltage_pu = 1.0
        self.frequency_hz = 50.0
        self.load_mw = 8.0
        self.solar_mw = 5.0
        self.battery_soc = 0.5
        self.battery_mw = 0.0
        self.episode_reward = 0.0
        self._fault_isolated = False

    def reset(self, task: str = "load_balance") -> MicrogridObservation:
        self._reset_internal()
        cfg = TASK_CONFIGS.get(task, TASK_CONFIGS["load_balance"])
        self.task = task
        self.max_steps = cfg["max_steps"]
        self.fault_at_step = cfg["fault_at_step"]
        self.solar_variability = cfg["solar_variability"]
        self.load_variability = cfg["load_variability"]

        # Slightly randomize starting state for variety
        self.load_mw = 8.0 + random.uniform(-1.5, 1.5)
        self.solar_mw = 5.0 + random.uniform(-1.0, 1.0)
        self.battery_soc = random.uniform(0.4, 0.7)

        return self._build_obs()

    def step(self, action: MicrogridAction) -> dict:
        self.step_count += 1

        # Clamp actions
        battery_dispatch = max(-10.0, min(10.0, action.battery_dispatch))
        load_shed = max(0.0, min(1.0, action.load_shed))
        switch_cmd = int(action.switch_cmd)

        # Trigger fault
        if self.fault_at_step and self.step_count == self.fault_at_step:
            self.fault_active = True
            self.fault_segment = 1
            self._fault_isolated = False

        # Apply switch commands
        if switch_cmd == 1:
            self.seg1_energized = False
        elif switch_cmd == 2:
            self.seg2_energized = False
        elif switch_cmd == 3:
            # Only restore if fault is isolated or no fault
            if not self.fault_active or self._fault_isolated:
                self.seg1_energized = True
                self.seg2_energized = True

        # Check if fault is isolated (faulted segment disconnected)
        if self.fault_active and self.fault_segment == 1 and not self.seg1_energized:
            self.fault_active = False
            self._fault_isolated = True

        # Update solar
        delta_solar = random.uniform(-self.solar_variability, self.solar_variability) * 5.0
        self.solar_mw = max(0.5, min(10.0, self.solar_mw + delta_solar))

        # Update load
        delta_load = random.uniform(-self.load_variability, self.load_variability) * 4.0
        self.load_mw = max(2.0, min(15.0, self.load_mw + delta_load))

        # Effective load after shedding and segment isolation
        effective_load = self.load_mw * (1.0 - load_shed)
        if not self.seg1_energized:
            effective_load *= 0.65   # seg1 carries ~35% of load
        if not self.seg2_energized:
            effective_load *= 0.75   # seg2 carries ~25% of load

        # Battery dispatch
        self.battery_mw = battery_dispatch
        self.battery_soc -= battery_dispatch * 0.015
        self.battery_soc = max(0.0, min(1.0, self.battery_soc))
        if self.battery_soc <= 0.05:
            self.battery_mw = min(0.0, self.battery_mw)  # can't discharge when empty
        if self.battery_soc >= 0.98:
            self.battery_mw = max(0.0, self.battery_mw)  # can't charge when full

        # Net power balance
        net_balance = self.solar_mw + self.battery_mw - effective_load

        # Grid response (simplified droop)
        self.voltage_pu = 1.0 + net_balance * 0.015
        self.frequency_hz = 50.0 + net_balance * 0.08
        self.voltage_pu = max(0.80, min(1.20, self.voltage_pu))
        self.frequency_hz = max(48.0, min(52.0, self.frequency_hz))

        # Compute reward
        reward = self._compute_reward(net_balance, load_shed, effective_load)
        self.episode_reward = round(self.episode_reward + reward, 4)

        done = self.step_count >= self.max_steps

        obs = self._build_obs()
        obs.net_balance_mw = round(net_balance, 4)
        obs.battery_mw = round(self.battery_mw, 4)

        return {
            "observation": obs.model_dump(),
            "reward": round(reward, 4),
            "done": done,
            "info": {
                "step": self.step_count,
                "episode_reward": self.episode_reward,
                "task": self.task,
            }
        }

    def _compute_reward(self, net_balance: float, load_shed: float, effective_load: float) -> float:
        reward = 0.0

        if self.task == "load_balance":
            # Reward for minimizing imbalance
            balance_score = max(0.0, 1.0 - abs(net_balance) / 5.0)
            reward = balance_score
            reward -= load_shed * 0.4           # penalize unnecessary shedding
            reward -= 0.3 if self.fault_active else 0.0

        elif self.task == "fault_recovery":
            if self.fault_active:
                # Escalating penalty — longer fault goes unhandled, worse it gets
                fault_duration = self.step_count - self.fault_at_step
                reward = -0.4 - (fault_duration * 0.02)
                reward = max(reward, -0.8)  # cap at -0.8
            elif self._fault_isolated:
                # Just isolated this step — big bonus
                balance_score = max(0.0, 1.0 - abs(net_balance) / 5.0)
                reward = balance_score * 0.5 + 0.5  # isolation bonus
                self._fault_isolated = False
            else:
                # Fault resolved — reward for stability
                balance_score = max(0.0, 1.0 - abs(net_balance) / 5.0)
                v_dev = abs(self.voltage_pu - 1.0)
                stability = max(0.0, 1.0 - v_dev * 4.0)
                reward = balance_score * 0.5 + stability * 0.5
            reward -= load_shed * 0.3

        elif self.task == "optimal_dispatch":
            # Penalize voltage and frequency deviations
            v_dev = abs(self.voltage_pu - 1.0)
            f_dev = abs(self.frequency_hz - 50.0)
            stability = max(0.0, 1.0 - v_dev * 3.0 - f_dev * 0.4)
            # Operational cost
            cost = abs(self.battery_mw) * 0.015 + load_shed * 0.6
            reward = stability - cost
            reward = max(-1.0, min(1.0, reward))

        return round(reward, 6)

    def _build_obs(self) -> MicrogridObservation:
        return MicrogridObservation(
            voltage_pu=round(self.voltage_pu, 4),
            frequency_hz=round(self.frequency_hz, 4),
            load_mw=round(self.load_mw, 4),
            solar_mw=round(self.solar_mw, 4),
            battery_soc=round(self.battery_soc, 4),
            battery_mw=round(self.battery_mw, 4),
            net_balance_mw=round(self.solar_mw + self.battery_mw - self.load_mw, 4),
            fault_active=self.fault_active,
            fault_segment=self.fault_segment,
            seg1_energized=self.seg1_energized,
            seg2_energized=self.seg2_energized,
            step=self.step_count,
            max_steps=self.max_steps,
            task=self.task,
            episode_reward=round(self.episode_reward, 4),
        )

    def get_state(self) -> dict:
        return self._build_obs().model_dump()