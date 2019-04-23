import numpy as np

import torch
from torch import Tensor
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from collections import Iterable
from rlkit.torch.core import np_ify, torch_ify


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, torch.sqrt(sigma_squared)


def _mean_of_gaussians(mus, sigmas):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma = torch.sqrt(torch.mean(sigmas**2, dim=0))
    return mu, sigma


class ProtoAgent(nn.Module):

    def __init__(self,
                 z_dim,
                 nets,
                 use_ae=False,
                 **kwargs
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.use_ae = use_ae
        if not self.use_ae:
            self.task_enc, self.policy, self.qf1, self.qf2, self.vf = nets
        else:
            self.task_enc, self.policy, self.qf1, self.qf2, self.vf, self.gt_enc, self.gt_dec = nets
        self.target_vf = self.vf.copy()
        self.recurrent = kwargs['recurrent']
        self.reparam = kwargs['reparameterize']
        self.use_ib = kwargs['use_information_bottleneck']
        self.tau = kwargs['soft_target_tau']
        self.reward_scale = kwargs['reward_scale']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.det_z = False

        # initialize task embedding to zero
        # (task, latent dim)
        self.register_buffer('z', torch.zeros(1, z_dim))
        # for incremental update, must keep track of number of datapoints accumulated
        self.register_buffer('num_z', torch.zeros(1))

        # initialize posterior to the prior
        if self.use_ib:
            self.z_dists = [torch.distributions.Normal(ptu.zeros(self.z_dim), ptu.ones(self.z_dim))]

    def clear_z(self, num_tasks=1):
        # if self.use_ib:
        #     self.z_dists = [torch.distributions.Normal(ptu.zeros(self.z_dim), ptu.ones(self.z_dim)) for _ in range(num_tasks)]
        #     z = [d.rsample() for d in self.z_dists]
        #     self.z = torch.stack(z)
        # else:
        #     self.z = self.z.new_full((num_tasks, self.z_dim), 0)
        # self.z_dim = None
        mu = ptu.zeros(num_tasks, self.z_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.z_dim)
        else:
            var = ptu.zeros(num_tasks, self.z_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        self.task_enc.reset(num_tasks) # clear hidden state in recurrent case

    def detach_z(self):
        self.z = self.z.detach()
        if self.recurrent:
            self.task_enc.hidden = self.task_enc.hidden.detach()

    def infer_gt_z(self, gammas):
        """

        :param gammas: mb,dim
        :return: mb,z_dim
        """
        return self.gt_enc(gammas) #

    def rec_gt_gamma(self, zs):
        """

        :param zs: mb,z_dim
        :return: mb,dim
        """
        return self.gt_dec(zs)

    def update_context(self, inputs):
        ''' update task embedding with a single transition '''
        # TODO there should be one generic method for preparing data for the encoder!!!
        o, a, r, no, d = inputs
        if self.sparse_rewards:
            r = ptu.sparsify_rewards(r)
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        # TODO: we can make this a bit more efficient by simply storing the natural params of the current posterior and add the new sample to update
        # then in the info bottleneck, we compute the the normal after computing the mean/variance from the natural params stored
        data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else: self.context = torch.cat([self.context, data], dim=1)
        # self.update_z(data)

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.task_enc(context)
        params = params.view(context.size(0), -1, self.task_enc.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.z_dim]
            sigma_squared = F.softplus(params[..., self.z_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def information_bottleneck(self, z):
        # assume input and output to be task x batch x feat
        mu = z[..., :self.z_dim]
        sigma_squared = F.softplus(z[..., self.z_dim:])
        z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        if not self.det_z:
            z_dists = [torch.distributions.Normal(m, s) for m, s in z_params]
            self.z_dists = z_dists
            z = [d.rsample() for d in z_dists]
        else:
            z = [p[0] for p in z_params]
        z = torch.stack(z)
        return z

    def compute_kl_div(self, mean=None):
        if mean is None:
            mean = ptu.zeros(self.z_dim)
        prior = torch.distributions.Normal(mean, ptu.ones(self.z_dim))
        kl_divs = [torch.distributions.kl.kl_divergence(z_dist, prior) for z_dist in self.z_dists]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    # TODO replace all usage of this to infer posterior
    def set_z(self, in_, idx):
        ''' compute latent task embedding only from this input data '''
        new_z = self.task_enc(in_)
        new_z = new_z.view(in_.size(0), -1, self.task_enc.output_size)
        if self.use_ib:
            new_z = self.information_bottleneck(new_z)
        else:
            new_z = torch.mean(new_z, dim=1)
        self.z = new_z

# not used
    # def update_z(self, in_):
    #     '''
    #     update current task embedding
    #      - by running mean for prototypical encoder
    #      - by updating hidden state for recurrent encoder
    #     '''
    #     z = self.z
    #     num_z = self.num_z
    #
    #     # TODO this only works for single task (t == 1)
    #     new_z = self.task_enc(in_)
    #     if new_z.size(0) != 1:
    #         raise Exception('incremental update for more than 1 task not supported')
    #     if self.recurrent:
    #         z = new_z
    #     else:
    #         new_z = new_z[0] # batch x feat
    #         num_updates = new_z.size(0)
    #         for i in range(num_updates):
    #             num_z += 1
    #             z += (new_z[i][None] - z) / num_z
    #     if self.use_ib:
    #         z = self.information_bottleneck(z)

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z#.unsqueeze(0) # 1,1,d
        obs = ptu.from_numpy(obs[None])
        in_ = (obs, z)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.tau)

    def forward(self, obs, actions, next_obs, enc_data, idx):
        # self.set_z(enc_data, idx)
        self.infer_posterior(enc_data)
        return self.infer(obs, actions, next_obs)

    def infer(self, obs, actions, next_obs, task_z=None):
        '''
        compute predictions of SAC networks for update

        regularize encoder with reward prediction from latent task embedding

        the returned task_z is txb,dim
        '''

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        if task_z is None:
            task_z = self.z
            task_z = [z.repeat(b, 1) for z in task_z]
            task_z = torch.cat(task_z, dim=0)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1 = self.qf1(obs, actions, task_z)
        q2 = self.qf2(obs, actions, task_z)
        v = self.vf(obs, task_z.detach())

        # run policy, get log probs and new actions
        in_ = (obs, task_z.detach())#torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=self.reparam, return_log_prob=True)

        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        return q1, q2, v, policy_outputs, target_v_values, task_z

    def min_q(self, obs, actions, task_z):
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)

        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    @property
    def networks(self):
        return [self.task_enc, self.policy, self.qf1, self.qf2, self.vf, self.target_vf]

class NewAgent(ProtoAgent):
    def __init__(self, explorer, seq_max_length, env, **kwargs):
        super().__init__(**kwargs)
        # self.seq_encoder = seq_encoder
        self.explorer = explorer
        self.envs = env
        self.seq_max_length = seq_max_length

    def set_z(self, in_, indices):
        """
        sequentially set z
        :param in_: does not need new data at all
        :return:
        """
        # TODO Attention: there is no need for data here
        # s,a,ns,r = in_
        # zs = list()
        # z_dists = list()
        # this means for each train step, the z is reproduced
        if not isinstance(indices, Iterable): indices = (indices,)

        s = [self.envs[i].reset_task(idx) for i,idx in enumerate(indices)] # a list of vectors
        new_z = None
        for _ in range(self.seq_max_length):
            s = ptu.from_numpy(np.asarray(s)) # mb,dim
            # a should be mb,1,dim
            a,np_a = self.explorer.get_actions((s,new_z), reparameterize=self.reparam) # the situation where new_z is None is handled inside explorer.forward()
            ns,r,term,env_info = zip(*[self.envs[i].step(ai) for i,ai in enumerate(np_a)])
            r = ptu.from_numpy(np.asarray(r).reshape((-1,1)))
            # inp = (s, a, r)
            new_z = self.task_enc(s,a,r) # mb,dim
            s = ns

            if True in term: break # TODO currently break once there is any terminal
            # zs.append(new_z)
            # z_dists.append(self.task_enc.z_dists[0]) # the sequential encoder by default processes one z at one time
        self.z = new_z # mb,zdim
        self.z_dists = self.task_enc.z_dists
        # Attention here !
        self.task_enc.clear() # clear the z of seq_encoder

    def forward(self, obs, actions, next_obs, enc_data, idx):
        # TODO: the current issue is that in algo it uses indices, but currently we only support single
        self.set_z(enc_data, idx)
        return self.infer(obs, actions, next_obs)





