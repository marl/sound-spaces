#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import logging
import math

import habitat
from habitat import Config, Dataset
from ss_baselines.common.baseline_registry import baseline_registry


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="AudioNavRLEnv")
class AudioNavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._continuous = config.CONTINUOUS

        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset) # passing TASK_CONFIG

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        logging.debug(super().current_episode)

        if self._continuous:
            self._previous_target_distance = self._distance_target()
        else:
            self._previous_target_distance = self.habitat_env.current_episode.info[
                "geodesic_distance"
            ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')

        assert not math.isnan(reward)

        return reward

    def _distance_target(self):
        return self._env.get_metrics()['distance_to_goal']

    def _episode_success(self):
        if self._env.task.is_stop_called and \
                ((self._continuous and self._distance_target() < self._success_distance) or
                 (not self._continuous and self._env.sim.reaching_goal)):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id
