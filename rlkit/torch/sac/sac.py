from collections import OrderedDict
import numpy as np
import pickle

import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_ify, torch_ify
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import MetaTorchRLAlgorithm
from rlkit.torch.sac.proto import ProtoAgent
import time,os
step = 0

class ProtoSoftActorCritic(MetaTorchRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            agent,
            explorer=None,
            # use_explorer=False,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            explorer_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            reparameterize=True,
            use_information_bottleneck=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=agent, # take only the agent, as self.policy, self.exploration_policy
            explorer=explorer,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )
        deterministic_embedding=False
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.eval_statistics = None
        self.kl_lambda = kl_lambda

        self.reparameterize = reparameterize
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards

        # TODO consolidate optimizers!
        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.agent.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.agent.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.agent.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.agent.task_enc.parameters(),
            lr=context_lr,
        )
        # if self.use_explorer:
        #     self.explorer_optimizer = optimizer_class(
        #         self.explorer.parameters(),
        #         lr=explorer_lr,
        #     )
        if self.use_explorer:
            self.exp_optimizer = optimizer_class(
                self.explorer.policy.parameters(),
                lr=policy_lr,
            )
            self.qf1exp_optimizer = optimizer_class(
                self.explorer.qf1.parameters(),
                lr=qf_lr,
            )
            self.qf2exp_optimizer = optimizer_class(
                self.explorer.qf2.parameters(),
                lr=qf_lr,
            )
            self.vfexp_optimizer = optimizer_class(
                self.explorer.vf.parameters(),
                lr=vf_lr,
            )


    def sample_data(self, indices, encoder=False):
        # sample from replay buffer for each task
        # TODO(KR) this is ugly af
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            if encoder:
                batch = self.get_encoding_batch(idx=idx)
            else:
                batch = self.get_batch(idx=idx)
            o = batch['observations'][None, ...]
            a = batch['actions'][None, ...]
            r = batch['rewards'][None, ...]
            no = batch['next_observations'][None, ...]
            t = batch['terminals'][None, ...]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            terms.append(t)
        obs = torch.cat(obs, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_obs = torch.cat(next_obs, dim=0)
        terms = torch.cat(terms, dim=0)
        return [obs, actions, rewards, next_obs, terms]

    def prepare_encoder_data(self, obs, act, rewards):
        ''' prepare task data for encoding '''
        # for now we embed only observations and rewards
        # assume obs and rewards are (task, batch, feat)
        if self.sparse_rewards:
            rewards = ptu.sparsify_rewards(rewards)
        task_data = torch.cat([obs, act, rewards], dim=2)
        return task_data

    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        batch = self.sample_data(indices, encoder=True)

        # zero out context and hidden encoder state
        if self.use_explorer: self.explorer.clear_z(num_tasks=len(indices))
        else: self.agent.clear_z(num_tasks=len(indices))

        for i in range(num_updates):
            # TODO(KR) argh so ugly
            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in batch]
            obs_enc, act_enc, rewards_enc, nobs_enc, _ = mini_batch
            if not self.use_explorer: nobs_enc = None
            self._take_step(indices, obs_enc, act_enc, rewards_enc, nobs_enc)

            # stop backprop
            self.agent.detach_z()

    def optimize_q(self, qf1_optimizer, qf2_optimizer, rewards, num_tasks, terms, target_v_values, q1_pred, q2_pred):
        # qf and encoder update (note encoder does not get grads from policy or vf)
        qf1_optimizer.zero_grad()
        qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        error1 = (q1_pred - q_target)
        error2 = (q2_pred - q_target)
        qf_loss = torch.mean(error1**2) + torch.mean(error2**2)
        qf_loss.backward()
        # self.writer.add_scalar('qf', qf_loss, step)
        qf1_optimizer.step()
        qf2_optimizer.step()
        return error1,error2,qf_loss
        # context_optimizer.step()

    def optimize_p(self, vf_optimizer, agent, policy_optimizer, obs, new_actions, task_z, log_pi, v_pred, policy_mean, policy_log_std, pre_tanh_value):
        # compute min Q on the new actions
        min_q_new_actions = self.agent.min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        vf_optimizer.zero_grad()
        vf_loss.backward()
        # self.writer.add_scalar('vf', vf_loss, step)
        vf_optimizer.step()
        agent._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        if self.reparameterize:
            policy_loss = (
                    log_pi - log_policy_target + v_pred.detach()  # to make it around 0
            ).mean()
        else:
            policy_loss = (
                    log_pi * (log_pi - log_policy_target + v_pred).detach()
            ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        # pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        # if self.use_explorer:
        #     self.explorer_optimizer.zero_grad()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        return log_policy_target, vf_loss, policy_loss

    def _take_step(self, indices, obs_enc, act_enc, rewards_exp, nobs_enc):
        global step

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)
        enc_data = self.prepare_encoder_data(obs_enc, act_enc, rewards_exp)

        # run inference in networks
        q1_pred, q2_pred, v_pred, policy_outputs, target_v_values, task_z = self.agent(obs, actions, next_obs, enc_data, indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        pre_tanh_value = policy_outputs[-1]
        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        error1,error2,qf_loss = self.optimize_q(self.qf1_optimizer, self.qf2_optimizer,
                        rewards, num_tasks, terms, target_v_values, q1_pred, q2_pred)
        self.context_optimizer.step()
        log_pi_target,vf_loss, policy_loss = self.optimize_p(self.vf_optimizer, self.agent, self.policy_optimizer,
                        obs, new_actions, task_z, log_pi, v_pred, policy_mean, policy_log_std, pre_tanh_value)
        # self.writer.add_histogram('act_adv', log_pi - log_pi_target + v_pred, step)
        # self.writer.add_histogram('logp',log_pi, step)
        # self.writer.add_scalar('qf', qf_loss, step)
        # self.writer.add_scalar('vf',vf_loss, step)
        if self.use_explorer:
            task_z = task_z.detach()
            self.explorer.z = task_z[::self.batch_size]
            q1_exp, q2_exp, v_exp, exp_outputs, target_v_exp, _ = self.explorer.infer(obs_enc, act_enc, nobs_enc, task_z=task_z)
            exp_actions, exp_mean, exp_log_std, exp_log_pi = exp_outputs[:4]
            exp_tanh_value = exp_outputs[-1]
            rewards_exp = -(error1 + error2) * self.exp_error_scale / 2. # small as possible
            rewards_exp = rewards_exp.detach()
            _,_,qf_exp = self.optimize_q(self.qf1exp_optimizer, self.qf2exp_optimizer,
                            rewards_exp, num_tasks, terms, target_v_exp, q1_exp, q2_exp)
            exp_logp_target,vf_exp,exp_loss = self.optimize_p(self.vfexp_optimizer, self.agent, self.exp_optimizer,
                            obs_enc, exp_actions, task_z, exp_log_pi, v_exp, exp_mean, exp_log_std, exp_tanh_value)
            if step%20==0:
                self.writer.add_histogram('exp_adv', exp_log_pi - exp_logp_target + v_exp, step)
                self.writer.add_histogram('logp_exp', exp_log_pi,step)
            self.writer.add_scalar('qf_exp', qf_exp,step)
            self.writer.add_scalar('vf_exp', vf_exp,step)

        # if self.use_explorer:
        #     self.explorer_optimizer.step()

        # self.writer.add_scalar('actor', policy_loss, step)
        # self.writer.add_histogram('logp', log_pi, step)
        # self.writer.add_histogram('adv', log_pi - log_policy_target + v_pred, step)

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            # TODO this is kind of annoying and higher variance, why not just average
            # across all the train steps?
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_dists[0].mean)))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_dists[0].variance))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
        step += 1
    # TODO: this may cause problem if is used in the future, remember to make agent arg.
    def sample_z_from_prior(self):
        self.agent.clear_z()

    def sample_z_from_posterior(self, agent, idx, eval_task=False):
        batch = self.get_encoding_batch(idx=idx, eval_task=eval_task)
        obs = batch['observations'][None, ...]
        act = batch['actions'][None, ...]
        rewards = batch['rewards'][None, ...]
        in_ = self.prepare_encoder_data(obs, act, rewards)
        # TODO: the sequential does not need a replay buffer actually
        agent.set_z(in_, idx)

    @property
    def networks(self):
        ret = self.agent.networks + [self.agent]
        if self.use_explorer: ret = ret + self.explorer.networks + [self.explorer]
        return ret

    # def get_epoch_snapshot(self, epoch):
    #     snapshot = OrderedDict(
    #         qf1=self.qf1.state_dict(),
    #         qf2=self.qf2.state_dict(),
    #         policy=self.agent.policy.state_dict(),
    #         vf=self.vf.state_dict(),
    #         target_vf=self.target_vf.state_dict(),
    #         context_encoder=self.agent.context_encoder.state_dict(),
    #     )
    #     return snapshot
