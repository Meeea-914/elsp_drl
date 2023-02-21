from typing import Tuple, Callable, Optional, List, Union, Dict, Type
import gym
import torch.nn as nn
import torch as th
from modules.set_transformer.models import StandardSelfAttention
from experiment.env_create import get_last_env
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.nn import Linear
import torch.nn.functional as F


def prod_feature_dim():
    return get_last_env().scheduling_unit.scheduling.prod_feature_dim


def env_feature_dim():
    return get_last_env().scheduling_unit.scheduling.env_feature_dim


# self attention
class Net3(nn.Module):

    def __init__(self, feature_dim, last_layer_dim_pi, last_layer_dim_vf):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.feature_dim = feature_dim
        self.layer = nn.Sequential()  # nn.ModuleDict(), nn.ModuleList()
        self.layer.add_module('transformer1', StandardSelfAttention(prod_feature_dim(), output_dim_multiple=4))
        self.layer.add_module('transformer2', StandardSelfAttention(self.layer.transformer1.output_dim, output_dim_multiple=4))
        self.layer.add_module('p_l_r1', Linear(prod_feature_dim() + env_feature_dim() + self.layer.transformer2.output_dim, 64))
        self.layer.add_module('p_l_r2', Linear(64, self.latent_dim_pi))
        self.layer.add_module('v_l_r1', Linear(prod_feature_dim() + env_feature_dim() + self.layer.transformer2.output_dim, 64))
        self.layer.add_module('v_l_r2', Linear(64, self.latent_dim_vf))

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        s1, s2 = get_last_env().decode_features(features)
        s2_ = th.stack([s2 for i in range(s1.shape[1])], dim=1)
        set_out = F.softplus(self.layer.transformer2(self.layer.transformer1(s1)))
        # print(s2_, s1, set_out)
        # print(s2_.shape, s1.shape, set_out.shape)
        s_as = th.cat((s2_, s1, set_out), dim=2)
        s = (s_as.mean(dim=1))
        # print(s, s_as)
        ps = []
        for d1 in range(s_as.shape[1]):
            s_a = s_as[:, d1:d1 + 1, :].squeeze(1)
            p_r1 = F.softplus(self.layer.p_l_r1(s_a))
            p_r2 = F.softplus(self.layer.p_l_r2(p_r1))
            ps.append(p_r2)
        p = th.stack(ps, dim=1)

        v_r1 = F.softplus(self.layer.v_l_r1(s))
        v_r2 = F.softplus(self.layer.v_l_r2(v_r1))
        return p, v_r2


class ExtractorForSSA(nn.Module):

    def __init__(self, feature_dim, last_layer_dim_pi, last_layer_dim_vf):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.feature_dim = feature_dim
        self.layer = nn.Sequential()  # nn.ModuleDict(), nn.ModuleList()
        self.layer.add_module('transformer1', StandardSelfAttention(prod_feature_dim(), hidden_dim=32, head_num=2))
        self.layer.add_module('transformer2', StandardSelfAttention(self.layer.transformer1.output_dim, hidden_dim=64, head_num=2))
        self.layer.add_module('p_l_r1', Linear(prod_feature_dim() + env_feature_dim() + self.layer.transformer2.output_dim, 64))
        self.layer.add_module('p_l_r2', Linear(64, self.latent_dim_pi))
        self.layer.add_module('v_l_r1', Linear(prod_feature_dim() + env_feature_dim() + self.layer.transformer2.output_dim, 64))
        self.layer.add_module('v_l_r2', Linear(64, self.latent_dim_vf))

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        s1, s2 = get_last_env().decode_features(features)
        s2_ = th.stack([s2 for i in range(s1.shape[1])], dim=1)
        set_out = F.softplus(self.layer.transformer2(self.layer.transformer1(s1)))
        # print(s2_, s1, set_out)
        # print(s2_.shape, s1.shape, set_out.shape)
        s_as = th.cat((s2_, s1, set_out), dim=2)
        s = (s_as.mean(dim=1))
        # print(s, s_as)
        ps = []
        for d1 in range(s_as.shape[1]):
            s_a = s_as[:, d1:d1 + 1, :].squeeze(1)
            p_r1 = F.softplus(self.layer.p_l_r1(s_a))
            p_r2 = F.softplus(self.layer.p_l_r2(p_r1))
            ps.append(p_r2)
        p = th.stack(ps, dim=1)

        v_r1 = F.softplus(self.layer.v_l_r1(s))
        v_r2 = F.softplus(self.layer.v_l_r2(v_r1))
        return p, v_r2


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
        if 'extractor' in kwargs:
            self.extractor_class = kwargs['extractor']
            kwargs.pop('extractor')
        else:
            self.extractor_class = Net3
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
        self.mlp_extractor = self.extractor_class(self.features_dim, 16, 16)


if __name__ == "__main__":
    """"""
