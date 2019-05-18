import numpy as np

import torch
from torch import Tensor
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.util import random_choice
from collections import Iterable
from rlkit.torch.core import np_ify, torch_ify


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    # sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
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
                 confine_num_c=False,
                 dif_policy=0,
                 **kwargs
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.use_ae = use_ae
        num_base_net = 4
        self.task_enc, self.policy, self.rew_func, self.vf = nets[:num_base_net]
        self.dif_policy = dif_policy
        if len(nets)==num_base_net+1: self.gt_dec = nets[-1]
        elif len(nets)==num_base_net+2:
            if self.use_ae: self.gt_enc, self.gt_dec = nets[-2:]
            else: self.gt_dec,self.ci2gam = nets[-2:]

        self.target_vf = self.vf.copy()
        self.recurrent = kwargs['recurrent']
        self.reparam = kwargs['reparameterize']
        self.use_ib = kwargs['use_information_bottleneck']
        self.tau = kwargs['soft_target_tau']
        self.reward_scale = kwargs['reward_scale']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.det_z = False
        self.context = None
        self.confine_num_c = confine_num_c
        self.num_updates = None
        # initialize task embedding to zero
        # (task, latent dim)
        self.register_buffer('z', torch.zeros(1, z_dim))
        self.register_buffer('z_means', torch.zeros(1, z_dim))
        self.register_buffer('z_vars', torch.zeros(1, z_dim))
        # for incremental update, must keep track of number of datapoints accumulated
        # self.register_buffer('num_z', torch.zeros(1))

        # initialize posterior to the prior
        # if self.use_ib:
        #     self.z_dists = [torch.distributions.Normal(ptu.zeros(self.z_dim), ptu.ones(self.z_dim))]

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

    def trans_z(self, mu, var, deterministic=False):
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z(deterministic=deterministic)

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
        dim=3
        ndim = o.ndim
        if isinstance(r, np.float64): r = np.array([r])
        for _ in range(dim-ndim):
            o,a,r = o[None,...],a[None,...],r[None,...]
        o = ptu.from_numpy(o)
        a = ptu.from_numpy(a)
        r = ptu.from_numpy(r)
        # TODO: we can make this a bit more efficient by simply storing the natural params of the current posterior and add the new sample to update
        # then in the info bottleneck, we compute the the normal after computing the mean/variance from the natural params stored
        data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else: self.context = torch.cat([self.context, data], dim=1)
        # self.update_z(data)

    def infer_posterior(self, context, infer_freq=0, deterministic=False):
        ''' compute q(z|c) as a function of input context and sample new z from it
        return nb,zdim if inferfreq==0 else nb,nupdate,zdim
        '''
        if self.confine_num_c: context = random_choice(context)
        params = self.task_enc(context)
        params = params.view(context.size(0), -1, self.task_enc.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        # if self.use_ib:
        mu = params[..., :self.z_dim]
        sigma_squared = F.softplus(params[..., self.z_dim:])
        if infer_freq==0:
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            mp, sp = zip(*z_params)
            self.z_means = torch.stack(mp)# b,num,zdim
            self.z_vars = torch.stack(sp)
        else:
            z_mparams_list, z_sparams_list = list(),list()

            num = context.size(1)
            start = 0
            m = torch.unbind(mu) # ntask, nsample, dim
            s = torch.unbind(sigma_squared)

            while start<num:
                # alist of b list of mean and variance
                # if start>0:
                #     z_params = [_product_of_gaussians(torch.cat((mm[start:min(num,start+infer_freq)],zm)),
                #                                       torch.cat((ss[start:min(num,start+infer_freq)],zs)))
                #                 for mm, ss,zm,zs in zip(m,s,z_mean,z_var)]
                # else:
                #     z_params = [_product_of_gaussians(mm[start:min(num,start+infer_freq)], ss[start:min(num,start+infer_freq)]) for mm, ss in zip(m,s)]
                z_params = [_product_of_gaussians(mm[:min(num,start+infer_freq)], ss[:min(num,start+infer_freq)]) for mm, ss in zip(m,s)]
                # mp2,sp2 = zip(*z_params2)
                mp,sp = zip(*z_params)
                z_mean = torch.stack(mp) # ntask,dim
                z_var = torch.stack(sp) # ntask,dim
                z_mparams_list.append(z_mean)# a list of a list of b tasks'z posterior params
                z_sparams_list.append(z_var)
                # z_mean = torch.unbind(z_mean.view(-1,1,self.z_dim))
                # z_var = torch.unbind(z_var.view(-1,1,self.z_dim))
                start += infer_freq
            # # numupdate,b,zdim > nb,zdim
            if self.num_updates is None: self.num_updates = len(z_mparams_list)
            self.z_means = torch.cat(z_mparams_list)#.view(-1,self.z_dim)
            self.z_vars = torch.cat(z_sparams_list)#.view(-1,self.z_dim)

        # sum rather than product of gaussians structure
        # else:
        #     self.z_means = torch.mean(params, dim=1)
        self.sample_z(deterministic=deterministic)

    def sample_z(self, batch=None, z_means=None, deterministic=False):
        if z_means is None:
            z_means = self.z_means
            z_vars = self.z_vars
        else:
            z_vars = torch.stack([ptu.ones(self.z_dim) for _ in range(z_means.size(0))])
        if self.use_ib:
            if deterministic:
                if batch is None: self.z = self.z_means
                else: self.z = self.z_means.view(-1,1,self.z_dim).repeat(1,batch,1) # ntask,bs,dim
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
            if batch is None:
                z = [d.rsample() for d in posteriors]
                self.z = torch.stack(z)
            else:
                z = [torch.stack([d.rsample() for _ in range(batch)]) for d in posteriors]
                return torch.stack(z) # mb,b,dim

        else:
            self.z = self.z_means

    # def information_bottleneck(self, z):
    #     # assume input and output to be task x batch x feat
    #     mu = z[..., :self.z_dim]
    #     sigma_squared = F.softplus(z[..., self.z_dim:])
    #     z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
    #     if not self.det_z:
    #         z_dists = [torch.distributions.Normal(m, s) for m, s in z_params]
    #         self.z_dists = z_dists
    #         z = [d.rsample() for d in z_dists]
    #     else:
    #         z = [p[0] for p in z_params]
    #     z = torch.stack(z)
    #     return z
    # TODO: there was bug when computing p(z|gam,c) and p(z|c)
    def compute_kl_div(self, mean=None):
        z_dists = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in
                      zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        if mean is None: # div between p(z|c) and N(0,1)
            posteriors = z_dists
            mean = ptu.zeros(self.z_dim)
            prior = torch.distributions.Normal(mean, ptu.ones(self.z_dim))
            kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        else:
            posteriors = [torch.distributions.Normal(m,ptu.ones(self.z_dim)) for m in mean]
            priors = z_dists
            kl_divs = [torch.distributions.kl.kl_divergence(prior, post) for prior,post in zip(priors,posteriors)]
        kl_divs = torch.mean(torch.stack(kl_divs),dim=-1, keepdim=True) # mb,1
        kl_div_sum = torch.sum(kl_divs)
        return kl_divs, kl_div_sum

    def compute_kl_div2(self, large_c, small_c):
        """

        :param large_c: (mean, var)
        :param small_c:
        :return:
        """
        p1 = [torch.distributions.Normal(m,v) for m,v in zip(*large_c)]
        p2 = [torch.distributions.Normal(m, v) for m, v in zip(*small_c)]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post,prior in zip(p1,p2)]
        kl_divs = torch.mean(torch.stack(kl_divs),dim=-1, keepdim=True) # mb,1
        kl_div_sum = torch.sum(kl_divs)
        return kl_divs, kl_div_sum

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

    def forward(self, obs, actions, next_obs, enc_data, idx=None, infer_freq=0):
        """

        :param obs: ntask,bs,dim
        :param actions: ntask,bs,dim
        :param next_obs: ntask,bs,dim
        :param enc_data:
        :param idx:
        :param infer_freq:
        :return: rew: ntask,bs,1
        """
        # self.set_z(enc_data, idx)
        self.infer_posterior(enc_data, infer_freq) # from context to z posterior and sample
        return self.infer(obs, actions, next_obs, infer_freq=infer_freq)

    def pred_cur_rew(self, obs, actions, task_z=None):
        ntask, bs, _ = obs.size()
        # obs = obs.view(ntask * bs, -1)
        # actions = actions.view(ntask * bs, -1)
        if task_z is None:
            task_z = self.z  # ntask,z_dim
            task_z = task_z.unsqueeze(-2)  # ntask,1,z_dim
            task_z = task_z.repeat(1, bs, 1)#ntask,bs,dim .view(-1, self.z_dim)  # ntask*bs
        return self.rew_func(obs,actions, task_z)

    def infer(self, obs, actions, next_obs, task_z=None, infer_freq=0):
        """

        :param obs: ntask,bs,dim
        :param actions: ntask,bs,dim
        :param next_obs: ntask,bs,dim
        :param task_z:
        :param infer_freq:
        :return: rew: ntask,bs,1
        """

        ntask, bs, _ = obs.size()
        if infer_freq==0:
            obs = obs.view(ntask * bs, -1)
            actions = actions.view(ntask * bs, -1)
            next_obs = next_obs.view(ntask * bs, -1)
            if task_z is None:
                task_z = self.z # ntask,z_dim
                task_z = task_z.unsqueeze(-2) # ntask,1,z_dim
                task_z = task_z.repeat(1,bs,1).view(-1,self.z_dim) # ntask*bs
        else:
            if task_z is None: task_z = self.z.view(self.num_updates, -1, self.z_dim)
            obs = obs.unsqueeze(0) # nupdate,ntask,bs,dim
            obs = obs.repeat(task_z.size(0),1,1,1)
            actions = actions.unsqueeze(0)
            actions = actions.repeat(task_z.size(0),1,1,1)
            next_obs = next_obs.unsqueeze(0)
            next_obs = next_obs.repeat(task_z.size(0), 1, 1, 1)
            task_z = task_z.unsqueeze(2)
            task_z = task_z.repeat(1,1,bs,1) # nupdate,ntask,bs,dim
            # task_z = [z.repeat(bs, 1) for z in task_z]
            # task_z = torch.cat(task_z, dim=0)

        # cur_pred_rew = self.rew_func(obs, actions, task_z)
        # cur_pred_rew = cur_pred_rew.view(ntask,bs,1)
        v = self.vf(obs, task_z.detach())

        in_ = (obs, task_z.detach())#torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=self.reparam, return_log_prob=True)

        return v, policy_outputs, task_z
    # TODO get rid of min_q
    def q_func(self, obs, actions, task_z, discount, terms):
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(-1, actions.size(-1))
        task_z = task_z.unsqueeze(1).repeat(1,b,1).view(-1,self.z_dim)
        terms = terms.view(-1,1)

        cur_rew = self.rew_func(obs, actions, task_z)
        ns_ret = self.vf(obs, task_z)

        return cur_rew+(1-terms)*discount*ns_ret

    @property
    def networks(self):
        return [self.task_enc, self.policy, self.rew_func, self.vf]

# class NewAgent(ProtoAgent):
#     def __init__(self, explorer, seq_max_length, env, **kwargs):
#         super().__init__(**kwargs)
#         # self.seq_encoder = seq_encoder
#         self.explorer = explorer
#         self.envs = env
#         self.seq_max_length = seq_max_length
#
#     def set_z(self, in_, indices):
#         """
#         sequentially set z
#         :param in_: does not need new data at all
#         :return:
#         """
#         # TODO Attention: there is no need for data here
#         # s,a,ns,r = in_
#         # zs = list()
#         # z_dists = list()
#         # this means for each train step, the z is reproduced
#         if not isinstance(indices, Iterable): indices = (indices,)
#
#         s = [self.envs[i].reset_task(idx) for i,idx in enumerate(indices)] # a list of vectors
#         new_z = None
#         for _ in range(self.seq_max_length):
#             s = ptu.from_numpy(np.asarray(s)) # mb,dim
#             # a should be mb,1,dim
#             a,np_a = self.explorer.get_actions((s,new_z), reparameterize=self.reparam) # the situation where new_z is None is handled inside explorer.forward()
#             ns,r,term,env_info = zip(*[self.envs[i].step(ai) for i,ai in enumerate(np_a)])
#             r = ptu.from_numpy(np.asarray(r).reshape((-1,1)))
#             # inp = (s, a, r)
#             new_z = self.task_enc(s,a,r) # mb,dim
#             s = ns
#
#             if True in term: break # TODO currently break once there is any terminal
#             # zs.append(new_z)
#             # z_dists.append(self.task_enc.z_dists[0]) # the sequential encoder by default processes one z at one time
#         self.z = new_z # mb,zdim
#         self.z_dists = self.task_enc.z_dists
#         # Attention here !
#         self.task_enc.clear() # clear the z of seq_encoder
#
#     def forward(self, obs, actions, next_obs, enc_data, idx):
#         # TODO: the current issue is that in algo it uses indices, but currently we only support single
#         self.set_z(enc_data, idx)
#         return self.infer(obs, actions, next_obs)





