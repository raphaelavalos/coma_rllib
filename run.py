import ray
from ray import tune
from ray.tune.registry import register_trainable, register_env
from smac_rllib import RLlibStarCraft2Env

from coma.trainer import COMATrainer

if __name__ == '__main__':
    ray.init(address='auto')
    env_config = {
        'map_name': '3m',
        "state_last_action": False,
    }

    def smac_env_creator(args):
        return RLlibStarCraft2Env(**args)

    env = smac_env_creator(env_config)
    horizon = env.horizon
    env.close()
    del env

    register_env('smac', smac_env_creator)
    register_trainable('coma', COMATrainer)
    gamma = 0.99
    config = {
        "use_coma": True,
        "lambda": 0.8,
        'seed': 1,
        "gamma": gamma,
        "min_iter_time_s": 1,
        "communication": False,
        "horizon": horizon,
        "num_episodes": 8,
        "grad_clip": 10,
        'target_network_update_freq': 200,
        "rollout_fragment_length": 1,
        "_use_trajectory_view_api": True,
        "batch_mode": "complete_episodes",
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
            "max_seq_len": horizon,

        },
        "actor_lr": 5e-4,
        "critic_lr": 5e-4,
        'framework': 'torch',
        'env': 'smac',
        'num_workers': 0,
        'num_cpus_per_worker': 2,
        'num_envs_per_worker': 1,
        'env_config': env_config,
        'num_gpus': 0.0,
        "entropy_coeff": 0.00,
        "tau": 1,

    }

    tune.run(COMATrainer,
                 name='coma_smac',
                 config=config,
                 metric='episode_reward_mean',
                 mode='max',
                 )