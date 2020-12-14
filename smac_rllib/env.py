# Adapted from https://github.com/oxwhirl/smac

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym.spaces import MultiDiscrete, Box, Dict
from smac.env import StarCraft2Env


class RLlibStarCraft2Env:
    """Wraps a smac StarCraft env to be compatible with RLlib multi-agent."""

    def __init__(self, **smac_args):
        """Create a new multi-agent StarCraft env compatible with RLlib.

        Arguments:
            smac_args (dict): Arguments to pass to the underlying
                smac.env.starcraft.StarCraft2Env instance.

        Examples:
            >>> from smac_rllib import RLlibStarCraft2Env
            >>> env = RLlibStarCraft2Env(map_name="8m")
            >>> print(env.reset())
        """

        self._env = StarCraft2Env(**smac_args)
        self.horizon = self._env.episode_limit
        self.nbr_agents = self._env.n_agents
        self._ready_agents = []
        self.observation_space = Dict({
            "obs": Box(-1, 1, shape=(self.nbr_agents, self._env.get_obs_size(),)),
            "avail_actions": Box(0, 1, shape=(self.nbr_agents, self._env.get_total_actions(),)),
            "state": Box(-float('inf'), float('inf'), shape=(self._env.get_state_size(),)),
            "battle_won": Box(0,1, shape=(1,), dtype=np.bool),
            "dead_allies": Box(0,self.nbr_agents, shape=(1,), dtype=np.int),
            "dead_enemies": Box(0, int(1e3), shape=(1,), dtype=np.int)
        })
        self.action_space = MultiDiscrete([self._env.get_total_actions()] * self.nbr_agents)

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """

        self._env.reset()
        return self._observe()

    def _observe(self, info=None):
        if info is None:
            info = {}
        dead_allies = sum([agent.health == 0 for agent in self._env.agents.values()])
        dead_enemies = sum([agent.health == 0 for agent in self._env.enemies.values()])
        return {
            "obs": np.stack(self._env.get_obs(), axis=0),
            "avail_actions": np.array(self._env.get_avail_actions(),  dtype=np.float),
            "state": self._env.get_state(),
            "battle_won": np.array([info.get('battle_won', False)]),
            "dead_allies": np.array([dead_allies]),
            "dead_enemies": np.array([dead_enemies]),
        }

    def step(self, action_list):
        rew, done, info = self._env.step(action_list)
        return self._observe(info), rew, done, info

    def close(self):
        """Close the environment"""
        self._env.close()

    def __del__(self):
        self.close()

