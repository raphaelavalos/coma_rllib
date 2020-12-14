from typing import Dict, List

import gym
import numpy as np
from gym.spaces import Box, MultiDiscrete
from ray.rllib import SampleBatch
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import \
    add_time_dimension as add_time_dimension_
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_torch, TensorType
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.typing import ModelConfigDict

torch, nn = try_import_torch()


def add_time_dimension(padded_inputs: TensorType,
                       *,
                       max_seq_len: int,
                       framework: str = "tf",
                       time_major: bool = False):
    inputs = add_time_dimension_(padded_inputs,
                                 max_seq_len=max_seq_len,
                                 framework=framework,
                                 time_major=False)
    if time_major:
        inputs = inputs.permute(
            (1, 0) + tuple(range(2, len(inputs.size())))).contiguous()
    return inputs


class COMATorchModel(TorchModelV2, nn.Module):
    """Extension of standard TorchModelV2 to COMA."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str = 'COMATorchModel',
                 communication: bool = True):
        nn.Module.__init__(self)
        super(COMATorchModel, self).__init__(obs_space, action_space,
                                             num_outputs,
                                             model_config, name)

        self.communication = communication
        assert self.is_time_major()
        self.recurrent = True

        if hasattr(self.obs_space, "original_space") and isinstance(
                self.obs_space.original_space, gym.spaces.Dict):
            original_space = self.obs_space.original_space
            self.has_avail_actions = 'avail_actions' in original_space.spaces
            self.has_real_state = 'state' in original_space.spaces
            self.has_q_value = 'q_value' in original_space.spaces
            self.has_value = 'value' in original_space.spaces
            self.true_obs_space = original_space['obs']
            if self.has_real_state:
                self.state_space = original_space['state']
        else:
            self.has_real_state = False
            self.has_q_value = False
            self.has_value = False
            self.state_space = None
            self.offsets = None
            self.true_obs_space = self.obs_space

        if not isinstance(self.true_obs_space, Box):
            raise UnsupportedSpaceException(
                "Space {} is not supported as observation.".format(
                    self.true_obs_space)
            )

        if not isinstance(action_space, MultiDiscrete):
            raise UnsupportedSpaceException(
                "Space {} is not supported as action.".format(self.action_space)
            )

        assert len(
            self.true_obs_space.shape) == 2, "Observation space is supposed " \
                                             "to have 2 dimensions."

        self.nbr_agents = self.true_obs_space.shape[0]
        self.nbr_actions = int(self.action_space.nvec[0])
        self.gru_cell_size = model_config.get("gru_cell_size")

        self.inference_view_requirements.update(
            {
                SampleBatch.OBS: ViewRequirement(shift=0),
                SampleBatch.PREV_ACTIONS: ViewRequirement(SampleBatch.ACTIONS,
                                                          space=action_space,
                                                          shift=-1),
                SampleBatch.ACTIONS: ViewRequirement(space=action_space),
                "state_in_{}".format(0): ViewRequirement(
                    "state_out_{}".format(0),
                    space=Box(-1.0, -1.0,
                              shape=(self.nbr_agents, self.gru_cell_size)),
                    shift=-1)
            }
        )

        self.stage1, self.gru, self.stage2 = self.create_actor()
        self.critic = self.create_critic()
        self.target_critic = self.create_critic()
        self.target_critic.load_state_dict(self.critic.state_dict())

    def create_actor(self):
        model_config = self.model_config
        layers = []
        activation_stage1 = model_config.get("fcnet_activation_stage1")
        hiddens_stage1 = model_config.get("fcnet_hiddens_stage1")

        self.gru_cell_size = model_config.get("gru_cell_size")
        activation_stage2 = model_config.get("fcnet_activation_stage2")
        hiddens_stage2 = model_config.get("fcnet_hiddens_stage2")

        prev_layer_size = self.true_obs_space.shape[1]  # obs
        prev_layer_size += self.nbr_agents  # one hot encoding of the agent id

        for size in hiddens_stage1:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation_stage1))
            prev_layer_size = size

        stage1 = nn.Sequential(*layers)

        gru = nn.GRU(
            input_size=prev_layer_size,
            hidden_size=self.gru_cell_size,
            num_layers=1,
            batch_first=not self.is_time_major()
        )

        prev_layer_size = self.gru_cell_size

        layers = []
        for size in hiddens_stage2:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation_stage2))
            prev_layer_size = size

        layers.append(
            SlimFC(
                in_size=prev_layer_size,
                out_size=self.nbr_actions,
                initializer=normc_initializer(1.0)
            )
        )

        stage2 = nn.Sequential(*layers)
        return stage1, gru, stage2

    def create_critic(self):
        layers = []
        input_size = np.prod(self.true_obs_space.shape)
        if self.has_real_state:
            input_size += np.prod(self.state_space.shape)
        input_size += self.nbr_agents
        input_size += 2 * self.nbr_agents * self.nbr_actions
        prev_layer_size = input_size
        activation = self.model_config['fcnet_activation_critic']
        for size in self.model_config['fcnet_hiddens_critic']:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(1.0)
                )
            )
            prev_layer_size = size

        layers.append(SlimFC(
            in_size=prev_layer_size,
            out_size=self.nbr_actions,
            initializer=normc_initializer(1.0)
        ))
        return nn.Sequential(*layers)

    def q_values(self, input_dict: Dict[str, TensorType],
                 target: bool = False) -> TensorType:
        nbr_agents = self.nbr_agents
        input_dict['obs'] = restore_original_dimensions(input_dict['obs'],
                                                        self.obs_space,
                                                        self.framework)
        obs = input_dict[SampleBatch.OBS]['obs']
        B = obs.shape[0]
        obs = obs.view(B, -1)

        if self.has_real_state:
            state = input_dict[SampleBatch.OBS]['state']
            obs = torch.cat([state, obs], dim=-1)
        obs = obs.view(B, 1, -1)
        obs = obs.expand(-1, nbr_agents, -1)
        agent_indexes = torch.eye(n=nbr_agents, dtype=obs.dtype,
                                  device=obs.device)
        agent_indexes.unsqueeze_(0)
        agent_indexes = agent_indexes.expand(B, -1, -1)

        actions = input_dict[SampleBatch.ACTIONS]
        if self.communication:
            actions = actions.view((B, nbr_agents, nbr_agents))[:, :, 0]
        actions = nn.functional.one_hot(actions, self.nbr_actions)
        actions = actions.unsqueeze(1).repeat(1, nbr_agents, 1, 1)
        agent_mask = (1 - torch.eye(self.nbr_agents, dtype=actions.dtype,
                                    device=actions.device))
        agent_mask = agent_mask.unsqueeze(0).unsqueeze(-1)
        actions = actions * agent_mask
        actions = actions.to(obs.dtype)
        actions = actions.view((B, nbr_agents, -1))

        prev_actions = input_dict[SampleBatch.PREV_ACTIONS]
        prev_actions = nn.functional.one_hot(prev_actions, self.nbr_actions)
        prev_actions = prev_actions.unsqueeze(1).repeat(1, nbr_agents, 1, 1)
        prev_actions = prev_actions.to(obs.dtype)
        prev_actions = prev_actions.view((B, nbr_agents, -1))

        x = torch.cat([obs, agent_indexes, prev_actions, actions], dim=-1)
        mlp = self.critic if not target else self.target_critic
        x = mlp(x)  # B, nbr_agents, nbr_actions

        if self.has_avail_actions:
            avail_actions = input_dict["obs"]["avail_actions"]
            inf_mask = torch.clamp(torch.log(avail_actions), FLOAT_MIN,
                                   FLOAT_MAX)
            x = x + inf_mask

        return x

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType], seq_lens: TensorType) -> (
            TensorType, List[TensorType]):
        nbr_agents = self.nbr_agents
        cell_size = self.gru_cell_size
        obs = input_dict[SampleBatch.OBS]['obs']
        B = obs.shape[0]
        h = state[0]
        R = h.shape[0]
        max_T = seq_lens.max().item()
        obs = add_time_dimension(obs, max_seq_len=max_T,
                                 framework=self.framework,
                                 time_major=self.is_time_major())

        agent_indexes = torch.eye(n=nbr_agents, dtype=h.dtype,
                                  device=h.device).unsqueeze(0).unsqueeze(0)
        agent_indexes = agent_indexes.expand(max_T, R, -1, -1)
        x = torch.cat([obs, agent_indexes], dim=-1)
        x = self.stage1(x)
        x = x.view(max_T, R * self.nbr_agents, -1)
        h = h.view(1, R * self.nbr_agents, cell_size)
        mems, h = self.gru(x, h)
        h = h.view(R, nbr_agents, cell_size)
        mems = mems.view(max_T, R, nbr_agents, cell_size)

        output = self.stage2(mems)

        if self.has_avail_actions:
            avail_actions = add_time_dimension(
                input_dict['obs']['avail_actions'], max_seq_len=max_T,
                framework=self.framework, time_major=self.is_time_major())
            avail_actions = avail_actions.view(max_T, R, nbr_agents,
                                               self.nbr_actions)
            inf_mask = torch.clamp(torch.log(avail_actions), FLOAT_MIN,
                                   FLOAT_MAX)
            output = output + inf_mask
        output = output.view(B, self.num_outputs)

        return output, [h, ]

    @override(TorchModelV2)
    def get_initial_state(self) -> List[np.ndarray]:
        h = [
            np.zeros((self.nbr_agents, self.gru_cell_size), dtype=np.float32),
        ]
        return h

    def policy_variables(self, as_dict: bool = False):
        if as_dict:
            dict_ = {
                **self.stage1.state_dict(),
                **self.gru.state_dict(),
            }
            if self.stage2:
                dict_.update(self.stage2.state_dict())
            return dict_
        return list(self.stage1.parameters()) + \
               list(self.gru.parameters()) + \
               (list(self.stage2.parameters()) if self.stage2 else list())

    def critic_variables(self, as_dict: bool = False):
        if as_dict:
            return self.critic.state_dict()
        return list(self.critic.parameters())

    def epsilon_value(self) -> float:
        return self.epsilon_scheduler(self.nbr_episodes)
