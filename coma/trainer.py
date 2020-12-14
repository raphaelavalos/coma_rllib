import logging
import time
from typing import List

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation import collect_metrics
from ray.rllib.execution.common import _check_sample_batch_type, \
    _get_shared_metrics, SAMPLE_TIMER, NUM_TARGET_UPDATES
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.utils.typing import SampleBatchType

from coma.policy import COMATorchPolicy, EpsilonCOMA

logger = logging.getLogger(__name__)

LAST_ITER_UPDATE = "last_iter_update"
NUM_ITER_LOOP = "num_iter_loop"


def evaluate(trainer, worker_set):
    episodes = []
    for _ in range(trainer.config["evaluation_num_episodes"]):
        episodes.append(worker_set.local_worker().sample())
    metric = collect_metrics(worker_set.local_worker())
    metric['battle_won_per'] = np.mean(
        [episode['battle_won'][-1] for episode in episodes]).item()
    return metric


# yapf: disable
# --__sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # If false the algorithm changes to Independent Actor Critic.
    "use_coma": True,
    "lambda": .9,
    "gamma": .99,
    "communication": True,
    # Max global norm for each gradient calculated by worker
    "grad_clip": 40.0,
    'target_network_update_freq': 50,
    # ModelConf
    "rollout_fragment_length": 1,
    "num_episodes": 8,
    "_use_trajectory_view_api": True,
    "batch_mode": "complete_episodes",
    "nbr_agents": None,
    "model": {
        "use_lstm": True,
        '_time_major': True,
        "custom_model_config": {
            "gru_cell_size": 64,
            "fcnet_activation_stage1": "relu",
            "fcnet_activation_stage2": "relu",
            "fcnet_hiddens_stage1": [64, ],
            "fcnet_hiddens_stage2": [],
            "fcnet_hiddens_critic": [128, 128],
            "fcnet_activation_critic": "relu",
        },
        "max_seq_len": 200,

    },
    "exploration_config": {
        "type": EpsilonCOMA,
        "initial_epsilon": 0.5,
        "final_epsilon": 0.01,
        "epsilon_timesteps": int(100000)
    },
    "actor_lr": 5e-4,
    "critic_lr": 5e-4,
    # "log_level": "DEBUG",
    "framework": "torch",
    "reward_range": None,
    "entropy_coeff": 0.0,
    "tau": 1,
    "timesteps_per_iteration": 0,
    'num_workers': 0,
    'num_envs_per_worker': 1,
    'num_gpus': 0.,
    "custom_eval_function": evaluate,
    "evaluation_interval": 100,
    "evaluation_num_episodes": 200,
    "evaluation_config": {
        "explore": False,
    },
})


def validate_config(config):
    pass


class ConcatEpisodes:
    """Callable used to merge episodes for training.

    Examples:
        >>> rollouts = ConcatEpisodes(...)
        >>> rollouts = rollouts.combine(ConcatEpisodes(num_episodes=8))
        >>> print(next(rollouts).count)
        8
    """

    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.buffer = []
        self.count = 0
        self.batch_start_time = None

    def _on_fetch_start(self):
        if self.batch_start_time is None:
            self.batch_start_time = time.perf_counter()

    def __call__(self, batch: SampleBatchType) -> List[SampleBatchType]:
        _check_sample_batch_type(batch)
        self.buffer.append(batch)
        self.count += 1
        if self.count >= self.num_episodes:
            out = SampleBatch.concat_samples(self.buffer)
            timer = _get_shared_metrics().timers[SAMPLE_TIMER]
            timer.push(time.perf_counter() - self.batch_start_time)
            timer.push_units_processed(self.count)
            self.batch_start_time = None
            self.buffer = []
            self.count = 0
            return [out]
        return []


class UpdateTargetNetwork:
    """Periodically call policy.update_target() on all trainable policies.

    This should be used with the .for_each() operator after training step
    has been taken.

    Examples:
        >>> train_op = UpdateTargetNetwork(...).for_each(TrainOneStep(...))
        >>> update_op = train_op.for_each(
        ...     UpdateTargetNetwork(workers, target_update_freq=500))
        >>> print(next(update_op))
        None

    Updates the LAST_ITER_UPDATE, NUM_ITER_LOOP and NUM_TARGET_UPDATES counters
    in the local iterator context. The value of the last update counter is used
    to track when we should update the target next.
    """

    def __init__(self,
                 workers,
                 target_update_freq,
                 policies=frozenset([])):
        self.workers = workers
        self.target_update_freq = target_update_freq
        self.policies = (policies or workers.local_worker().policies_to_train)
        self.metric = NUM_ITER_LOOP

    def __call__(self, _):
        metrics = _get_shared_metrics()
        metrics.counters[self.metric] += 1
        cur_ts = metrics.counters[self.metric]
        last_update = metrics.counters[LAST_ITER_UPDATE]
        if cur_ts - last_update >= self.target_update_freq:
            to_update = self.policies
            self.workers.local_worker().foreach_trainable_policy(
                lambda p, p_id: p_id in to_update and p.update_target())
            metrics.counters[NUM_TARGET_UPDATES] += 1
            metrics.counters[LAST_ITER_UPDATE] = cur_ts


def execution_plan(workers, config):
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    train_op = rollouts \
        .combine(ConcatEpisodes(
        num_episodes=config["num_episodes"])) \
        .for_each(TrainOneStep(workers)) \
        .for_each(UpdateTargetNetwork(
        workers, config['target_network_update_freq']))

    return StandardMetricsReporting(train_op, workers, config)


COMATrainer = build_trainer(
    name="COMA",
    default_config=DEFAULT_CONFIG,
    default_policy=COMATorchPolicy,
    validate_config=validate_config,
    execution_plan=execution_plan)
