import copy
from typing import Tuple, Dict, Union

import gym
import numpy as np
from ray.rllib import Policy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper, \
    TorchMultiCategorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration import EpsilonGreedy as EpsilonGreedy_
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import convert_to_torch_tensor, \
    convert_to_non_torch_type, sequence_mask, FLOAT_MIN
from ray.rllib.utils.typing import TrainerConfigDict, TensorType

import coma
from coma.model import COMATorchModel, add_time_dimension

torch, nn = try_import_torch()


class EpsilonCOMA(EpsilonGreedy_):
    """Implement the exploration strategy of COMA """

    def _get_torch_exploration_action(
            self, action_distribution: ActionDistribution, explore: bool,
            timestep: Union[int, TensorType]):
        logits = action_distribution.inputs
        self.last_timestep = timestep
        exploit = action_distribution.deterministic_sample()
        batch_size = logits.size()[0]
        nbr_agents = self.model.nbr_agents
        nbr_actions = self.model.nbr_actions

        action_logp = torch.zeros(batch_size, dtype=torch.float)

        # Explore.
        if explore:
            # Get the current epsilon.
            epsilon = self.epsilon_schedule(self.last_timestep)
            logits = logits.view(batch_size * nbr_agents, nbr_actions)
            proba = torch.nn.functional.softmax(logits, -1)
            new_proba = (1 - epsilon) * proba + epsilon / (
                    logits > FLOAT_MIN).sum(-1, keepdims=True).float()
            new_proba = torch.where(logits <= FLOAT_MIN, new_proba.new_zeros(1),
                                    new_proba)
            explore_action = torch.distributions.Categorical(
                probs=new_proba).sample().long()
            return explore_action.view(exploit.shape), action_logp
        # Return the deterministic "sample" (argmax) over the logits.
        else:
            return exploit, action_logp


class COMATorchDist(TorchMultiCategorical):
    def __init__(self, input, model):
        super(TorchMultiCategorical, self).__init__(input, model)

        inputs_split = self.inputs.split(tuple(model.action_space.nvec), dim=1)
        self.cats = [torch.distributions.categorical.Categorical(logits=input_)
                     for input_ in inputs_split]


def make_model_and_action_dist(policy: Policy,
                               observation_space: gym.spaces.Space,
                               action_space: gym.spaces.Space,
                               config: TrainerConfigDict) -> Tuple[
    ModelV2, TorchDistributionWrapper]:
    model_config = copy.deepcopy(config['model'])
    model_config.update(model_config.pop('custom_model_config'))
    model = COMATorchModel(observation_space,
                           action_space,
                           num_outputs=sum(action_space.nvec),
                           model_config=model_config,
                           communication=config.get('communication', True))

    return model, COMATorchDist


class TargetNetworkMixin:
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, config: TrainerConfigDict):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self, tau=None):
        tau = tau or self.config.get("tau")
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        # Full sync from Q-model to target Q-model.
        if tau == 1.0:
            self.model.target_critic.load_state_dict(
                self.model.critic.state_dict())
        # Partial (soft) sync using tau-synching.
        else:
            model_vars = list(self.model.critic.parameters())
            target_model_vars = list(self.model.target_critic.parameters())
            assert len(model_vars) == len(target_model_vars), \
                (model_vars, target_model_vars)
            for var, var_target in zip(model_vars, target_model_vars):
                var_target.data = tau * var.data + \
                                  (1.0 - tau) * var_target.data


def validate_spaces(policy: Policy, observation_space: gym.Space,
                    action_space: gym.Space,
                    config: TrainerConfigDict) -> None:
    if not isinstance(observation_space, gym.spaces.Box):
        raise UnsupportedSpaceException("Observation space must be a box.")

    # TODO: check the action_space nvec.


def make_coma_optimizers(policy: Policy, config: TrainerConfigDict):
    policy._actor_optimizer = torch.optim.RMSprop(
        params=policy.model.policy_variables(),
        lr=config['actor_lr'],
        alpha=0.99,
        eps=1e-8,
        weight_decay=0
    )

    policy._critic_optimizer = torch.optim.RMSprop(
        params=policy.model.critic_variables(),
        lr=config['critic_lr'],
        alpha=0.99,
        eps=1e-8,
        weight_decay=0
    )

    return policy._actor_optimizer, policy._critic_optimizer


def stats(policy, train_batch):
    return {
        **{
            "actor_loss": policy.actor_loss.item(),
            "q_value_loss": policy.critic_loss.item(),
            "entropy": policy.entropy.item(),
        },
        **policy.exploration.get_info(),
    }


def compute_target(policy,
                   sample_batch,
                   other_agent_batches=None,
                   episode=None):
    sample_batch_ = {key: sample_batch[key] for key in
                     [SampleBatch.OBS, SampleBatch.PREV_ACTIONS,
                      SampleBatch.ACTIONS]}
    sample_batch_ = convert_to_torch_tensor(sample_batch_, policy.device)

    next_obs = restore_original_dimensions(
        convert_to_torch_tensor(sample_batch[SampleBatch.NEXT_OBS]),
        policy.model.obs_space, policy.framework)
    sample_batch['battle_won'] = convert_to_non_torch_type(
        next_obs['battle_won'])
    target_q_values = policy.model.q_values(sample_batch_, target=True)
    target_q_values = convert_to_non_torch_type(target_q_values)
    actions = sample_batch[SampleBatch.ACTIONS]
    actions = actions.reshape(actions.shape[:1] + (policy.model.nbr_agents, -1))
    target = np.take_along_axis(target_q_values, actions, axis=-1)
    target = np.squeeze(target, -1)
    reward = np.expand_dims(sample_batch[SampleBatch.REWARDS], -1)
    gamma = policy.config['gamma']
    lambda_ = policy.config['lambda']
    y = np.zeros_like(target)
    y[-1] = reward[-1]
    for t in range(y.shape[0] - 2, -1, -1):
        y[t] = reward[t] + (1 - lambda_) * gamma * target[
            t + 1] + gamma * lambda_ * y[t + 1]

    sample_batch[Postprocessing.VALUE_TARGETS] = y

    return sample_batch


def loss_fn(policy: Policy, model: ModelV2,
            dist_class: TorchDistributionWrapper, sample_batch: SampleBatch):
    max_seq_len = sample_batch['seq_lens'].max().item()
    mask = sequence_mask(sample_batch['seq_lens'], max_seq_len,
                         time_major=model.is_time_major()).view((-1, 1))
    mean_reg = sample_batch['seq_lens'].sum() * model.nbr_agents
    actions = sample_batch['actions'].view(
        (sample_batch['actions'].shape[0], model.nbr_agents, -1))[:, :, :1].to(
        torch.long)
    actions = add_time_dimension(actions, max_seq_len=max_seq_len,
                                 framework='torch', time_major=True).reshape_as(
        actions)

    logits_pi, _ = model(sample_batch,
                         [sample_batch['state_in_0'], ],
                         sample_batch['seq_lens'])
    logits_pi = logits_pi.view((logits_pi.shape[0], model.nbr_agents, -1))
    logits_pi_action = logits_pi[:, :, :model.nbr_actions]
    log_pi_action = nn.functional.log_softmax(logits_pi_action, dim=-1)
    pi_action = torch.exp(log_pi_action)
    log_pi_action_selected = torch.gather(log_pi_action, -1, actions).squeeze(
        -1)

    q_values = model.q_values(sample_batch, target=False)
    q_values = add_time_dimension(q_values, max_seq_len=max_seq_len,
                                  framework="torch",
                                  time_major=True).reshape_as(q_values)
    q_values_selected = torch.gather(q_values, -1, actions).squeeze(-1)
    q_values_target = sample_batch[Postprocessing.VALUE_TARGETS]
    q_values_target = add_time_dimension(q_values_target,
                                         max_seq_len=max_seq_len,
                                         framework="torch",
                                         time_major=True).reshape_as(
        q_values_target)
    td_error = q_values_selected - q_values_target

    with torch.no_grad():
        coma_avg = q_values_selected - (pi_action * q_values).sum(-1)
    entropy = - (log_pi_action * pi_action).sum(-1)

    critic_loss = torch.pow(mask * td_error, 2.0)
    actor_loss = mask * coma_avg * log_pi_action_selected
    entropy = mask * entropy

    policy.actor_loss = - actor_loss.sum() / mean_reg
    policy.critic_loss = critic_loss.sum() / mean_reg
    policy.entropy = entropy.sum() / mean_reg

    pi_loss = policy.actor_loss - policy.config[
        'entropy_coeff'] * policy.entropy

    return pi_loss, policy.critic_loss


def view_requirements_fn(policy: Policy) -> Dict[str, ViewRequirement]:
    """Function defining the view requirements for training/postprocessing.

    These go on top of the Policy's Model's own view requirements used for
    the action computing forward passes.

    Args:
        policy (Policy): The Policy that requires the returned
            ViewRequirements.

    Returns:
        Dict[str, ViewRequirement]: The Policy's view requirements.
    """
    ret = {
        SampleBatch.NEXT_OBS: ViewRequirement(
            SampleBatch.OBS, shift=1, used_for_training=False),
        Postprocessing.ADVANTAGES: ViewRequirement(shift=0),
        Postprocessing.VALUE_TARGETS: ViewRequirement(shift=0),
    }
    return ret


def setup_late_mixins(policy: Policy, obs_space: gym.spaces.Space,
                      action_space: gym.spaces.Space,
                      config: TrainerConfigDict) -> None:
    """Call all mixin classes' constructors before SimpleQTorchPolicy
    initialization.

    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    TargetNetworkMixin.__init__(policy, obs_space, action_space, config)


def apply_grad_clipping(policy, optimizer, loss):
    info = {}
    if policy.config["grad_clip"]:
        for param_group in optimizer.param_groups:
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
            params = list(
                filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                grad_gnorm = nn.utils.clip_grad_norm_(
                    params, policy.config["grad_clip"])
                if isinstance(grad_gnorm, torch.Tensor):
                    grad_gnorm = grad_gnorm.cpu().numpy()
                info["grad_gnorm"] = grad_gnorm
    return info


COMATorchPolicy = build_torch_policy(
    name="COMATorchPolicy",
    make_model_and_action_dist=make_model_and_action_dist,
    postprocess_fn=compute_target,
    optimizer_fn=make_coma_optimizers,
    validate_spaces=validate_spaces,
    get_default_config=lambda: coma.trainer.DEFAULT_CONFIG,
    view_requirements_fn=view_requirements_fn,
    mixins=[TargetNetworkMixin, ],
    after_init=setup_late_mixins,
    extra_grad_process_fn=apply_grad_clipping,
    stats_fn=stats,
    loss_fn=loss_fn)
