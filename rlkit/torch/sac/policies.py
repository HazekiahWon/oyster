import numpy as np
import torch
from torch import nn as nn

from rlkit.core.util import Wrapper
from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp
from rlkit.torch.core import np_ify
from torch.nn import functional as F
from rlkit.torch.core import PyTorchModule
from torch.autograd import Variable
from rlkit.torch import pytorch_util as ptu

USE_CUDA = True

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            latent_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.latent_dim = latent_dim
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, obs, deterministic=False):

        outputs = self.forward(obs, deterministic=deterministic)[0]
        return np_ify(outputs)

    def forward(
            self,
            obs,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        obs = torch.cat(obs, dim=-1)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, q, ks):
        # seq_len = len(ks)

        h = self.attn(q) # mb,1,c -> mb,1,c
        ks = ks.permute(0,2,1) # mb,c,d

        energies = torch.bmm(h,ks) # mb,1,d
        return F.softmax(energies, dim=-1) # mb,1,d
        #
        # # Create variable to store attention energies
        # attn_energies = Variable(torch.zeros(seq_len))  # B x 1 x S
        # if USE_CUDA: attn_energies = attn_energies.cuda()
        #
        # # Calculate energies for each encoder output
        # for i in range(seq_len):
        #     attn_energies[i] = self.score(q, ks[i])
        #
        # # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        # return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy

class DecomposedPolicy(PyTorchModule, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """


    def construct_fc(self, dim1, dim2, init_w=1e-3):
        # init_w = self.init_w
        tmp = nn.Linear(dim1, dim2)
        tmp.weight.data.uniform_(-init_w, init_w)
        tmp.bias.data.uniform_(-init_w, init_w)
        return tmp

    def construct_hidden(self, in_size, next_size, hidden_init=ptu.fanin_init, b_init_value=0.1):
        fc = nn.Linear(in_size, next_size)
        hidden_init(fc.weight)
        fc.bias.data.fill_(b_init_value)
        return fc

    def __init__(
            self,
            obs_dim,
            z_dim,
            latent_dim,
            action_dim,
            anet_sizes,
            obs_nlayer=2,
            z_nlayer=2,
            eta_nlayer=2,
            num_expz=None,
            std=None,

            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__()
        # super().__init__(
        #     hidden_sizes,
        #     input_size=obs_dim,
        #     output_size=action_dim,
        #     init_w=init_w,
        #     **kwargs
        # )
        self.hidden_activation = F.relu
        self.latent_dim = latent_dim
        self.log_std = None
        self.std = std
        # self.init_w = init_w
        self.last_fc = self.construct_fc(latent_dim, action_dim)
        if std is None:
            # last_hidden_size = latent_dim
            # if len(hidden_sizes) > 0:
            #     last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = self.construct_fc(latent_dim, action_dim)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

        obs_fc = [self.construct_hidden(obs_dim, latent_dim)]
        z_fc = [self.construct_hidden(z_dim, latent_dim)]
        # a_fc = [self.construct_hidden(2*latent_dim, latent_dim)] # s+eta
        a_fc = []

        for i in range(obs_nlayer):
            obs_fc.append(self.construct_hidden(latent_dim,latent_dim))
        for i in range(z_nlayer):
            z_fc.append(self.construct_hidden(latent_dim, latent_dim))
        # specify the action network's net sizes
        ain = 2*latent_dim
        for nex in anet_sizes:
            a_fc.append(self.construct_hidden(ain, nex))
            ain = nex

        self.obs_fc = nn.ModuleList(obs_fc)
        self.z_fc = nn.ModuleList(z_fc)
        self.a_fc = nn.ModuleList(a_fc)

        if eta_nlayer is not None:
            eta_fc = [self.construct_hidden(2*latent_dim, latent_dim)]
            for i in range(eta_nlayer):
                eta_fc.append(self.construct_hidden(latent_dim, latent_dim))
            self.eta_fc = nn.ModuleList(eta_fc)
            self.use_atn = False
        else:
            self.wsw = nn.Sequential(nn.Linear(latent_dim, num_expz),
                                     # nn.BatchNorm1d(num_expz), # cannnot batch norm when batch=1
                                     nn.ReLU()
                                     )
            # self.bn = nn.BatchNorm1d(num_expz)

            self.atn = Attn('general', hidden_size=latent_dim)
            self.use_atn = True

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, obs, deterministic=False):
        outputs = self.forward(obs, deterministic=deterministic)[0]
        return np_ify(outputs)

    def atn_eta(self, z, obs):
        """

        :param z: mb*b,c
        :param obs: mb*b,c
        :return:
        """
        #######
        # atn
        #######
        # mb,b,c = z.shape
        # z = z.permute(0,2,1) # b,c,1
        zw = self.wsw(z)  # mb*b,d, no normalize cuz size(0)=1
        zw = zw.unsqueeze(1) # 1,d
        z = z.unsqueeze(-1) # c,1
        # not sure if there should be normalization.
        expanded_z = torch.bmm(z, zw)  # mb*b,c,d
        z = expanded_z.permute(0, 2, 1) # mb*b,d,c
        obs = obs.unsqueeze(1) # mb*b,1,c

        weights = self.atn(obs, z)  # b,1,d
        eta = torch.bmm(weights, z)  # b,1,c
        return eta.squeeze(1)

    def direct_eta(self, z, obs):
        #######
        # direct concat
        #######
        h = torch.cat((obs, z), dim=-1)
        for i, fc in enumerate(self.eta_fc):
            h = self.hidden_activation(fc(h))
        eta = h
        return eta

    def forward(
            self,
            inp,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        obs,z = inp
        #######
        # state -> state feature
        #######
        h = obs
        obs_ = obs
        for i, fc in enumerate(self.obs_fc):
            h = self.hidden_activation(fc(h))
        obs = h # latent dim
        #######
        # z -> z embed
        #######
        h = z
        for i, fc in enumerate(self.z_fc):
            h = self.hidden_activation(fc(h))
        z = h # latent dim

        #######
        # get eta
        #######
        # eta = self.atn_eta(z,obs) if self.use_atn else self.direct_eta(z,obs) # latent dim
        eta = z
        #######
        # p(a|s,eta)
        #######
        h = torch.cat((obs,eta), dim=-1)
        #######################################
        for i,fc in enumerate(self.a_fc): # as the tanhpolicy
            h = self.hidden_activation(fc(h)) # latent dim

        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )


class MakeDeterministic(Wrapper, Policy):
    def __init__(self, stochastic_policy):
        super().__init__(stochastic_policy)
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations,
                                                  deterministic=True)
