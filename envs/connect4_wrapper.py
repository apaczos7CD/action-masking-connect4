from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo.classic import connect_four_v3

class OneAgentVsRandomGym(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, my_agent="player_0", seed: int | None = None):
        self.my_agent = my_agent
        self._rng = np.random.RandomState(seed)
        self._pz = connect_four_v3.env()  # AEC env
        self.observation_space = spaces.Box(low=0, high=1, shape=(6, 7, 2), dtype=np.int8)
        self.action_space = spaces.Discrete(7)
        self._terminal_override = False  # terminal po nielegalnej akcji

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._terminal_override = False
        self._pz.reset(seed=seed)
        self._advance_to_my_turn()
        return self._obs(), {}


    def step(self, action: int):
        if self._terminal_override:
            raise RuntimeError("Env jest terminalny; wywołaj reset().")
        # 1) nielegalny ruch => natychmiastowa porażka −1
        mask = self.action_masks()
        if not bool(mask[action]):
            self._terminal_override = True
            info = {"illegal_move": True}

            return self._obs(), -1.0, True, False, info

        # 2) legalny ruch agenta
        self._pz.step(action)
        if self._episode_done():
            r = self._reward_from_terminal()
            return self._obs(), r, True, False, {}

        # 3) odpowiedź przeciwnika (uniform‑random‑legal)
        opp_action = self._rng.choice(np.flatnonzero(self._current_mask()))
        self._pz.step(opp_action)
        if self._episode_done():
            r = self._reward_from_terminal()
            return self._obs(), r, True, False, {}

        return self._obs(), 0.0, False, False, {}


    # ===== Narzędzia =====
    def _advance_to_my_turn(self):
        while self._pz.agent_selection != self.my_agent and not self._episode_done():
            a = self._rng.choice(np.flatnonzero(self._current_mask()))
            self._pz.step(a)


    def _current_mask(self):
        obs = self._pz.observe(self._pz.agent_selection)
        return obs["action_mask"].astype(bool)


    def action_masks(self):
        obs = self._pz.observe(self.my_agent)
        return obs["action_mask"].astype(bool)


    def _obs(self):
        raw = self._pz.observe(self.my_agent)["observation"]  # (6,7,2)
        return raw.astype(np.int8)


    def _episode_done(self) -> bool:
        return any(self._pz.terminations.values()) or any(self._pz.truncations.values()) or self._terminal_override


    def _reward_from_terminal(self) -> float:
        if self._terminal_override:
            return -1.0
        return float(self._pz.rewards[self.my_agent])