from typing import Tuple, Callable, Optional, List, Union, Dict, Type
import gym
import torch.nn as nn
import torch as th
from experiment.env_create import get_last_env
from modules.set_transformer.models import StandardSelfAttention
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.nn import Linear
import torch.nn.functional as F


class NetRummukainen(nn.Module):

    def __init__(self, feature_dim, last_layer_dim_pi, last_layer_dim_vf):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.feature_dim = feature_dim
        self.layer = nn.Sequential()  # nn.ModuleDict(), nn.ModuleList()
        self.product_num = product_num = len(get_last_env().get_probable_actions()) - 1
        encoder_out_put = 8 * product_num
        self.action_num = action_num = product_num + 1
        obs_dim = product_num + 1
        self.layer.add_module('encoder', Linear(obs_dim, encoder_out_put))
        for i in range(action_num):
            self.layer.add_module('shared_layer_%d' % i, Linear(obs_dim + encoder_out_put, last_layer_dim_pi))
        for i in range(action_num):
            self.layer.add_module('value_layer_%d' % i, Linear(obs_dim, last_layer_dim_vf))

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        index = features[:, 0:1]
        v_index = index
        for i in range(self.latent_dim_pi - 1):
            index = th.cat([index, features[:, 0:1]], dim=-1)
        for i in range(self.latent_dim_vf - 1):
            v_index = th.cat([v_index, features[:, 0:1]], dim=-1)
        index = index.long()
        v_index = v_index.long()
        index = index.unsqueeze(1)
        v_index = v_index.unsqueeze(1)
        state = th.cat([features[:, 0: 1], features[:, 2:]], dim=-1)

        encoding = th.relu(self.layer.encoder(state))
        shared_layer_input = th.cat([state, encoding], dim=-1)
        ps = []
        for i in range(self.action_num):
            ps.append(getattr(self.layer, 'shared_layer_%d' % i)(shared_layer_input))
        p = th.stack(ps, dim=1)
        p = th.gather(p, 1, index.long()).squeeze(1)
        # p = th.softmax(p, dim=-1).squeeze(1)

        vs = []
        for i in range(self.action_num):
            vs.append(getattr(self.layer, 'value_layer_%d' % i)(state))
        v = th.stack(vs, dim=1)
        v = th.gather(v, 1, v_index.long()).squeeze(1)
        # v = th.softmax(v, dim=-1).squeeze(1)
        return p, v


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.prob = 0
        self.coe = 1.6

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = NetRummukainen(self.features_dim, 4, 4)


if __name__ == "__main__":
    """"""
