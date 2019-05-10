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
from rlkit.torch.modules import LayerNorm
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
            lat_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        obs_dim += lat_dim # adding lat_dim is to match with embpolicy
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        # self.latent_dim = latent_dim
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
        # print([x.size() for x in obs])
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
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob
        )



class EmbPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            lat_dim,
            action_dim,
            emb_nlayer=1,
            emb_dim=32,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            emb_dim,# embedded dim is the real obs dim
            lat_dim,
            action_dim,
            **kwargs)
        self.emb_nlayer = emb_nlayer
        self.emb_dim = emb_dim
        self.layer_norms0 = list()
        ############# embed observation, obs_dim - obsemb_dim (direct) or z_dim (atn)
        latent_dim = emb_dim
        obs_fc = []
        ln_cnt = 0
        in_dim = obs_dim
        out_dim = latent_dim
        for i in range(emb_nlayer+1):
            obs_fc.append(self.construct_hidden(in_dim, out_dim))

            # TODO layer norm
            if self.layer_norm:
                ln = LayerNorm(out_dim)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms0.append(ln)
                ln_cnt += 1
            in_dim = out_dim
        # if self.use_atn: obs_fc.append(self.construct_hidden(latent_dim, obs_emb_dim))
        # else: obs_fc.append(self.construct_hidden(latent_dim, latent_dim))
        self.obs_fc = nn.ModuleList(obs_fc)

    def forward(
            self,
            obs,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        o,lat = obs
        for i, fc in enumerate(self.obs_fc):
            o = fc(o)
            if self.layer_norm and i<len(self.obs_fc)-1:
                o = self.layer_norms0[i](o)
            o = self.hidden_activation(o)
        in_ = (o, lat)
        return super().forward(in_, reparameterize, deterministic, return_log_prob)

    def construct_fc(self, dim1, dim2, init_w=1e-3):
        # init_w = self.init_w
        tmp = nn.Linear(dim1, dim2)
        tmp.weight.data.uniform_(-init_w, init_w)
        tmp.bias.data.uniform_(-init_w, init_w)
        return tmp

    def construct_hidden(self, in_size, next_size, b_init_value=0.1):
        fc = nn.Linear(in_size, next_size)
        self.hidden_init(fc.weight)
        fc.bias.data.fill_(b_init_value)
        return fc

class HierPolicy(PyTorchModule, ExplorationPolicy):
    def __init__(self, hpolicy, lpolicy):
        self.save_init_params(locals())
        super().__init__()
        self.hpolicy = hpolicy
        self.lpolicy = lpolicy

    def forward(
            self,
            obs,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        o,z = obs
        eta_outs = self.hpolicy(obs, reparameterize, deterministic, return_log_prob)
        eta = eta_outs[0]
        obs = (o,eta)
        a_outs = self.lpolicy(obs, reparameterize, deterministic, return_log_prob)
        return a_outs+eta_outs # tuple concat

    @torch.no_grad()
    def get_actions(self, obs, deterministic=False):
        outputs = self.forward(obs, deterministic=deterministic)[0]
        return np_ify(outputs)

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

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

class FAUCore:
    @staticmethod
    def v2l(key_feature, func):
        """
        visible to latent
        :param key_feature: mb,1,c
        :param func: transform k to d dim, mb,1,c => mb,1,d
        :return: mb,d,c
        """
        B, _,_ = key_feature.shape
        graph_adj_v2l = func(key_feature)  # mb,1,d
        latent_nodes = torch.bmm(graph_adj_v2l.reshape((B,-1,1)),key_feature) # mb,d,c

        return latent_nodes
    @staticmethod
    def l2l( a, b):
        """

        :param a: be the values exactly
        :param b:
        :return:
        """
        a_normalized = F.normalize(a, dim=-1) # mb,d,c
        b_normalized = F.normalize(b, dim=-1) # mb,d,c
        # Step2: latent-to-latent message passing
        ###### afm=ln \dot ln: d c . c d - d d
        ###### ln =afm \dot ln: d d . d c - d c => bt,d,c => b, td, c =>
        affinity_matrix = torch.bmm(a_normalized,
                                    b_normalized.permute(0, 2, 1))  # mb,d,d
        # import pdb; pdb.set_trace()
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)
        # latent_nodes = torch.bmm(affinity_matrix/self.latent_dim, latent_nodes)
        a = torch.bmm(affinity_matrix, a)  # d,c

        return a
    @staticmethod
    def l2v(query_feature, latent_nodes, func):
        """
        latent to visible
        :param query_feature: b,c,h,w
        :param latent_nodes: b,d,c
        :param func: transform q to d dim, b,c,hw => b,d,hw
        :return: b,c,h,w
        """
        # Step3: latent-to-visible message passing
        ###### gal=aff(q): c h w - d h w - d hw
        ###### vn =ln \dot gal: c d . d hw - c hw - c h w
        graph_adj_l2v = func(query_feature)  # mb,1,d
        B, _, _ = query_feature.shape
        graph_adj_l2v = F.normalize(graph_adj_l2v.view(B, 1, -1), dim=-1)  # mb,1,d

        visible_nodes = torch.bmm(graph_adj_l2v, latent_nodes) # mb,1,c
        return visible_nodes

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

    def construct_hidden(self, in_size, next_size, b_init_value=0.1):
        fc = nn.Linear(in_size, next_size)
        self.hidden_init(fc.weight)
        fc.bias.data.fill_(b_init_value)
        return fc

    def __init__(
            self,
            obs_dim,
            z_dim,
            # latent_dim,
            action_dim,
            anet_sizes,
            obs_emb_dim=16, # or 16
            eta_emb_dim=64,
            eta_dim=16, # because z is small?
            obs_nlayer=2,
            z_nlayer=2,
            eta_nlayer=2,
            num_expz=None,
            atn_type='low-rank',
            std=None,
            hidden_init=ptu.fanin_init,
            layer_norm=False,
            layer_norm_kwargs=None,
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
        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()
        self.layer_norm = layer_norm
        self.layer_norms = []
        self.hidden_activation = F.relu
        self.hidden_init = hidden_init
        # self.latent_dim = latent_dim
        self.log_std = None
        self.std = std
        # self.init_w = init_w
        self.last_fc = self.construct_fc(anet_sizes[-1], action_dim)
        self.use_atn = eta_nlayer is None
        self.lrank = atn_type == 'low-rank'
        if std is None:
            # last_hidden_size = latent_dim
            # if len(hidden_sizes) > 0:
            #     last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = self.construct_fc(anet_sizes[-1], action_dim)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        # if we only embed observation,
        # when using direct eta, the dimensions do not matter.
        # when using atn eta, must ensure the embedded observation to have the same dimension as the z
        ############# embed observation, obs_dim - obsemb_dim (direct) or z_dim (atn)
        latent_dim = obs_emb_dim
        obs_fc = [self.construct_hidden(obs_dim, latent_dim)]
        ln_cnt = 0
        for i in range(obs_nlayer):
            obs_fc.append(self.construct_hidden(latent_dim,latent_dim))
            # TODO layer norm
            if self.layer_norm:
                ln = LayerNorm(latent_dim)
                self.__setattr__("layer_norm{}".format(i), ln_cnt)
                self.layer_norms.append(ln)
                ln_cnt += 1
        # if self.use_atn: obs_fc.append(self.construct_hidden(latent_dim, obs_emb_dim))
        # else: obs_fc.append(self.construct_hidden(latent_dim, latent_dim))
        self.obs_fc = nn.ModuleList(obs_fc)
        ############## eta net
        ### direct eta: obsembdim+z_dim - etadim
        ### atn eta: 2*obsembdim - eta_dim
        eta_in = latent_dim+z_dim if not self.use_atn else 2*latent_dim
        latent_dim = eta_emb_dim
        if not self.use_atn:
            eta_fc = [self.construct_hidden(eta_in, latent_dim)]
            for i in range(eta_nlayer-1):
                eta_fc.append(self.construct_hidden(latent_dim, latent_dim))
            eta_fc.append(self.construct_hidden(latent_dim,eta_dim))
            self.eta_fc = nn.ModuleList(eta_fc)
            # self.use_atn = False
        else:
            if atn_type!='low-rank':
                # should be the same as the embedded observation
                z_fc = [self.construct_hidden(z_dim, obs_emb_dim)]
                for i in range(z_nlayer):
                    z_fc.append(self.construct_hidden(obs_emb_dim, obs_emb_dim))
                # specify the action network's net sizes
                self.z_fc = nn.ModuleList(z_fc)
                self.wsw = nn.Sequential(nn.Linear(obs_emb_dim, num_expz),
                                         # nn.BatchNorm1d(num_expz), # cannnot batch norm when batch=1
                                         nn.ReLU()
                                         )
                # self.bn = nn.BatchNorm1d(num_expz)

                self.atn = Attn('general', hidden_size=obs_emb_dim)
            else:
                self.pre_transform_o = self.construct_hidden(obs_emb_dim, eta_dim)
                self.pre_transform_z = self.construct_hidden(z_dim, eta_dim)
                self.k_lin = nn.Sequential(
                    self.construct_hidden(eta_dim, num_expz), # number of vectors in latent space
                    # nn.BatchNorm1d(num_expz),
                    nn.ReLU())
                self.q_lin = nn.Sequential(
                    self.construct_hidden(eta_dim, num_expz), # number of vectors in latent space
                    # nn.BatchNorm1d(num_expz),
                    nn.ReLU())
                self.q_lin2 = nn.Sequential(
                    self.construct_hidden(eta_dim, num_expz), # number of vectors in latent space
                    # nn.BatchNorm1d(num_expz),
                    nn.ReLU())
            # self.use_atn = True
        ######### action net
        ### eta_dim+obsemb_dim - action_dim
        a_fc = [] # s+eta
        ain = eta_dim + obs_emb_dim
        for nex in anet_sizes:
            a_fc.append(self.construct_hidden(ain, nex))
            ain = nex
        self.a_fc = nn.ModuleList(a_fc)

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
        #######
        # z -> z embed
        #######
        h = z
        for i, fc in enumerate(self.z_fc):
            h = self.hidden_activation(fc(h))
        z = h # latent dim
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

    def fau_atn(self, z, obs):
        """
        should ensure eta dim is the same as obs emb dim
        :param z:
        :param obs:
        :return:
        """
        #######
        # embed z to have same dim as obs
        #######
        obs = self.hidden_activation(self.pre_transform_o(obs)).unsqueeze(1) # mb,1,
        z = self.hidden_activation(self.pre_transform_z(z)).unsqueeze(1) # mb,1
        # query = obs
        keys = FAUCore.v2l(z, self.k_lin) # make z: dxc
        queries = FAUCore.v2l(obs, self.q_lin)
        aff = FAUCore.l2l(keys, queries) # dxd
        res = FAUCore.l2v(obs, aff, self.q_lin2) # as I think eta depends more on state: 1xc
        return res.squeeze(1) # mb,c

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
        # get eta
        #######
        eta = self.direct_eta(z,obs) if not self.use_atn else (self.fau_atn(z,obs) if self.lrank else self.atn_eta(z,obs)) # latent dim
        # eta = z
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
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob
        )


class BasePolicy(PyTorchModule, ExplorationPolicy):

    def construct_fc(self, dim1, dim2, init_w=1e-3):
        # init_w = self.init_w
        tmp = nn.Linear(dim1, dim2)
        tmp.weight.data.uniform_(-init_w, init_w)
        tmp.bias.data.uniform_(-init_w, init_w)
        return tmp

    def construct_hidden(self, in_size, next_size, b_init_value=0.1):
        fc = nn.Linear(in_size, next_size)
        self.hidden_init(fc.weight)
        fc.bias.data.fill_(b_init_value)
        return fc

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, obs, deterministic=False):
        outputs = self.forward(obs, deterministic=deterministic)[0]
        return np_ify(outputs)

class BNHierPolicy(BasePolicy):

    def __init__(
            self,
            obs_dim,
            z_dim,
            action_dim,
            obsemb_sizes,
            etanet_sizes,
            anet_sizes,
            obs_emb_dim, # or 16
            eta_dim, # because z is small
            sparse=False,
            std=None,
            hidden_init=ptu.fanin_init,
            layer_norm=False,
            layer_norm_kwargs=None,
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
        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()
        self.layer_norm = layer_norm
        self.layer_norms = []
        self.sparse = sparse
        self.hidden_activation = F.relu
        self.hidden_init = hidden_init
        # self.latent_dim = latent_dim
        self.log_std = None
        self.std = std
        # self.init_w = init_w
        self.last_fc = self.construct_fc(anet_sizes[-1], action_dim)

        if std is None:
            self.last_fc_log_std = self.construct_fc(anet_sizes[-1], action_dim)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

        if obs_emb_dim!=0:
            obs_fc = list()
            in_size = obs_dim
            for nex_size in obsemb_sizes:
                obs_fc.append(self.construct_hidden(in_size, nex_size))
                in_size = nex_size
            obs_fc.append(self.construct_hidden(in_size, obs_emb_dim))
            self.obs_fc = nn.ModuleList(obs_fc)
            self.use_obs_emb = True
        else:
            self.use_obs_emb = False
            obs_emb_dim = obs_dim

        in_size = obs_emb_dim+z_dim # in_size = obs emb
        eta_fc = list()

        for nex_size in etanet_sizes:
            eta_fc.append(self.construct_hidden(in_size, nex_size))
            in_size = nex_size
        eta_fc.append(self.construct_hidden(in_size,eta_dim))
        self.eta_fc = nn.ModuleList(eta_fc)

        a_fc = [] # s+eta
        in_size = eta_dim + obs_emb_dim
        for nex_size in anet_sizes:
            a_fc.append(self.construct_hidden(in_size, nex_size))
            in_size = nex_size
        self.a_fc = nn.ModuleList(a_fc)

    def direct_eta(self, obs,z):
        #######
        # direct concat
        #######
        h = torch.cat((obs, z), dim=-1)
        for i, fc in enumerate(self.eta_fc):
            h = self.hidden_activation(fc(h))
        eta = h
        return eta

    def sparse_eta(self, obs, z):
        h = torch.cat((obs, z), dim=-1)
        n = len(self.eta_fc)
        for i, fc in enumerate(self.eta_fc):
            h = fc(h)
            if i==n-1: h = torch.sigmoid(h)
            else: h = self.hidden_activation(h)
        return h

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
        if self.use_obs_emb:
            h = obs
            # obs_ = obs
            for i, fc in enumerate(self.obs_fc):
                h = self.hidden_activation(fc(h))
            obs = h # latent dim

        #######
        # get eta
        #######
        eta = self.sparse_eta(obs, z) if self.sparse else self.direct_eta(obs, z)

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
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob
        )

class MDNPolicy(BasePolicy):

    def __init__(
            self,
            obs_dim,
            z_dim,
            action_dim,
            obsemb_sizes,
            anet_sizes,
            obs_emb_dim, # or 16
            eta_dim, # because z is small
            n_component=20,
            sparse=False,
            std=None,
            hidden_init=ptu.fanin_init,
            layer_norm=False,
            layer_norm_kwargs=None,
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
        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()
        self.layer_norm = layer_norm
        self.layer_norms = []
        self.sparse = sparse
        self.hidden_activation = F.relu
        self.hidden_init = hidden_init
        # self.latent_dim = latent_dim
        self.log_std = None
        self.std = std
        # self.init_w = init_w
        self.last_fc = self.construct_fc(anet_sizes[-1], action_dim*n_component)
        self.action_dim = action_dim
        self.n_component = n_component

        if std is None:
            self.last_fc_log_std = self.construct_fc(anet_sizes[-1], action_dim*n_component)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        self.last_pi = self.construct_fc(anet_sizes[-1], n_component)

        obs_fc = list()
        in_size = obs_dim
        if obs_emb_dim!=0:
            for nex_size in obsemb_sizes:
                obs_fc.append(self.construct_hidden(in_size, nex_size))
                in_size = nex_size
            obs_fc.append(self.construct_hidden(in_size, obs_emb_dim))
            self.obs_fc = nn.ModuleList(obs_fc)
            self.use_obs_emb = True
        else:
            self.use_obs_emb = False
            obs_emb_dim = obs_dim

        in_size = obs_emb_dim+z_dim # in_size = obs emb

        a_fc = [] # s+eta
        # in_size = eta_dim + obs_emb_dim
        for nex_size in anet_sizes:
            a_fc.append(self.construct_hidden(in_size, nex_size))
            in_size = nex_size
        self.a_fc = nn.ModuleList(a_fc)

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
        if self.use_obs_emb:
            h = obs
            for i, fc in enumerate(self.obs_fc):
                h = self.hidden_activation(fc(h))
            obs = h # latent dim

        h = torch.cat((obs,z), dim=-1)
        #######################################
        for i,fc in enumerate(self.a_fc): # as the tanhpolicy
            h = self.hidden_activation(fc(h)) # latent dim

        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h) # n*actiondim
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std
        pi = F.softmax(self.last_pi(h), dim=-1) # b,n_componentwhat about using sigmoid?
        # mixture of several tanh normal
        log_prob = None

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
                # log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()
        pi = pi.view(-1,1,self.n_component)
        action = action.view(-1,self.n_component, self.action_dim)
        mean = mean.view(-1,self.n_component, self.action_dim)
        action = torch.bmm(pi, action).squeeze(1) # b,1,adim
        mean = torch.matmul(pi,mean).squeeze(1)
        # log_std do not know how to
        if log_prob is not None:
            log_prob = log_prob.view(-1,self.n_component,self.action_dim)
            log_prob = torch.exp(log_prob)
            log_prob = torch.bmm(pi, log_prob).squeeze(1)
            log_prob = torch.log(log_prob)
            log_prob = torch.sum(log_prob, dim=-1,keepdim=True)


        return (
            action, mean, log_std, log_prob
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
