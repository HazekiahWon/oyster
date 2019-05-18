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
            rec_lambda=1.,
            gam_rew_lambda=.5,
            eq_enc=False,
            rew_mode=False,
            sar2gam=False,
            dif_policy=0,
            inc_enc=0,
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
        self.onehot_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss(reduction='none')
        self.l2_reg_criterion = nn.MSELoss()

        self.eval_statistics = None
        self.kl_lambda = kl_lambda
        self.rec_lambda = rec_lambda
        self.gam_rew_lambda = gam_rew_lambda
        self.eq_enc = eq_enc
        self.rew_mode = rew_mode
        self.sar2gam = sar2gam
        self.dif_policy = dif_policy
        self.inc_enc = inc_enc
        self.reparameterize = reparameterize
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_ae = False
        if hasattr(self.agent, 'use_ae'):
            self.use_ae = self.agent.use_ae

        # TODO consolidate optimizers!
        if self.dif_policy==1:
            self.hpolicy_optimizer = optimizer_class(
                self.agent.policy.hpolicy.parameters(),
                lr=policy_lr,
            )
            self.lpolicy_optimizer = optimizer_class(
                self.agent.policy.lpolicy.parameters(),
                lr=policy_lr,
            )
        else:
            self.policy_optimizer = optimizer_class(
                self.agent.policy.parameters(),
                lr=policy_lr,
            )
        self.rew_optimizer = optimizer_class(
            self.agent.rew_func.parameters(),
            lr=qf_lr,
        )
        # self.qf2_optimizer = optimizer_class(
        #     self.agent.qf2.parameters(),
        #     lr=qf_lr,
        # )
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
        if self.use_ae or self.eq_enc:
            self.dec_optimizer = optimizer_class(
                self.agent.gt_dec.parameters(),
                lr=context_lr,
            )
        if self.use_ae:
            self.enc_optimizer = optimizer_class(
                self.agent.gt_enc.parameters(),
                lr=context_lr,
            )
        if self.sar2gam:
            self.dec_optimizer2 = optimizer_class(
                self.agent.ci2gam.parameters(),
                lr=context_lr*10,
            )
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

    def pretrain(self, n_iter=50000, n_task=16):

        if self.use_explorer and self.eq_enc and self.sar2gam:
            for it in range(n_iter):
                indices = np.random.choice(self.train_tasks, size=min(n_task,len(self.train_tasks)), replace=False)
                z_bs = 16
                gammas = self.make_variation(indices)
                data4enc = self.sample_data(indices, encoder=True, batchs=z_bs)  # 20 sample for each task?
                # TODO is it necessary to use s,a with r
                data4enc = self.prepare_encoder_data(*data4enc[:3])  # s,a,r
                task_gam = self.agent.ci2gam(data4enc)
                gammas_ = gammas.view(-1,1,self.gamma_dim).repeat(1, task_gam.size(1), 1)
                _, rec_loss2 = self.mse_crit(gammas_, task_gam)  # only used to train ci>gam
                self.writer.add_scalar('pretrain_ci2gam_loss', rec_loss2, it)
                self.dec_optimizer2.zero_grad()
                rec_loss2.backward()
                self.dec_optimizer2.step()

    def sample_data(self, indices, encoder=False, batchs=None):
        # sample from replay buffer for each task
        # TODO(KR) this is ugly af
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            if encoder:
                batch = self.get_encoding_batch(idx=idx, batch=batchs)
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

    def prepare_context(self, idx):
        ''' sample context from replay buffer and prepare it '''
        batch = ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, trajs=self.recurrent))
        obs = batch['observations'][None, ...]
        act = batch['actions'][None, ...]
        rewards = batch['rewards'][None, ...]
        context = self.prepare_encoder_data(obs, act, rewards)
        return context

    def _do_training(self, indices, gammas=None):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        batch = self.sample_data(indices, encoder=True)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))
        if self.use_explorer: self.explorer.clear_z(num_tasks=len(indices))

        for i in range(num_updates):
            # TODO(KR) argh so ugly
            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in batch]
            obs_enc, act_enc, rewards_enc, nobs_enc, terms_enc = mini_batch
            if not self.use_explorer: nobs_enc = None
            self._take_step(indices, obs_enc, act_enc, rewards_enc, nobs_enc, terms_enc, gammas)

            # stop backprop
            self.agent.detach_z()
            if self.use_explorer: self.explorer.detach_z()

    def optimize_q(self, qf1_optimizer, qf2_optimizer, rewards, num_tasks, terms, target_v_values, q1_pred, q2_pred, scale_reward=True, infer_freq=0):
        # qf and encoder update (note encoder does not get grads from policy or vf)
        # if return_loss:
        qf1_optimizer.zero_grad()
        qf2_optimizer.zero_grad()
        if infer_freq!=0:
            n_updates = target_v_values.size(0)
            rewards_flat = rewards.unsqueeze(0).repeat(n_updates, 1, 1, 1)
            terms_flat = terms.unsqueeze(0).repeat(n_updates, 1, 1, 1)
        else:
            rewards_flat = rewards.view(-1,1)
            terms_flat = terms.view(-1, 1)
        # scale rewards for Bellman update
        if scale_reward: rewards_flat = rewards_flat * self.reward_scale
        q_target = rewards_flat + (1 - terms_flat) * self.discount * target_v_values
        if infer_freq==0:
            q1_err,q1_loss = self.mse_crit(q1_pred, q_target.detach())
            q2_err, q2_loss = self.mse_crit(q2_pred, q_target.detach())
            # err: mb.b
            q1_err,q2_err = [torch.mean(err.view(-1,self.batch_size),dim=-1,keepdim=True) for err in (q1_err,q2_err)]
        else:
            q1_err = self.mse_crit(q1_pred, q_target.detach(), dim_start=2, return_red=False) # q1pred:13,5,256,1 > 13,5,1
            q2_err = self.mse_crit(q2_pred, q_target.detach(), dim_start=2, return_red=False)
            # if allow the loss include inferior version of z, may help the actor when inference budget is limited
            # as the q function is learned with inferior version of z
            # can add discounting factors
            q1_loss = torch.mean(q1_err)
            q2_loss = torch.mean(q2_err)
        return q1_err + q2_err, q1_loss + q2_loss

    def optimize_p(self, vf_optimizer, agent, policy_optimizer, obs, v_pred, pout, terms, dif_policy=0, alpha=1):
        act, a_mean, a_logstd, a_logp = pout[:4]
        if dif_policy==1: eta, e_mean, e_logstd, e_logp = pout[4:]
        # compute min Q on the new actions
        min_q_new_actions = agent.q_func(obs, act, agent.z.detach(), self.discount, terms) # make obs and act, t,256,dim
        ######### vf loss involves the gradients of
        # v
        ###########
        # TODO: a can possibly ingest a detached eta
        if dif_policy==1: a_logp += e_logp
        a_logp = a_logp*alpha
        v_target = min_q_new_actions - a_logp
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        vf_optimizer.zero_grad()
        vf_loss.backward()
        # self.writer.add_scalar('vf', vf_loss, step)
        vf_optimizer.step()
        # no need for a target network for q
        # agent._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions
        ##### policy loss involves the gradients of
        # actor, does not involve task encoder, q
        #############################
        if self.reparameterize:
            policy_loss = (
                    a_logp - log_policy_target  # to make it around 0
            ).mean()
        else:
            policy_loss = (
                    a_logp * (a_logp - log_policy_target + v_pred).detach()
            ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (a_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (a_logstd ** 2).mean()
        if dif_policy==1:
            mean_reg_loss = self.policy_mean_reg_weight * (e_mean ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (e_logstd ** 2).mean()
        # pre_tanh_value = policy_outputs[-1]
        # pre_activation_reg_loss = self.policy_pre_activation_weight * (
        #     (pre_tanh_value ** 2).sum(dim=1).mean()
        # )
        policy_reg_loss = mean_reg_loss + std_reg_loss# + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        # if self.use_explorer:
        #     self.explorer_optimizer.zero_grad()
        if dif_policy==1:
            [opt.zero_grad() for opt in policy_optimizer]
        else:
            policy_optimizer.zero_grad()
        policy_loss.backward()
        if dif_policy==1:
            [opt.step() for opt in policy_optimizer]
        else: policy_optimizer.step()

        return log_policy_target, vf_loss, policy_loss

    def mse_crit(self, x, y, dim_start=1, return_red=True):
        # make sure x,y has same dim
        shape = x.shape[:dim_start]
        nreduced = torch.mean((x-y)**2, dim=tuple(range(dim_start,len(x.shape)))).view(*shape,1) # mb,1
        if return_red:
            reduced = torch.mean(nreduced)
            return nreduced,reduced
        else: return nreduced


    def _take_step(self, indices, obs_enc, act_enc, rew_enc, nobs_enc, terms_enc, gammas=None):
        global step
        z_bs = 20
        num_tasks = len(indices)

        # sample data: o,a,r,no,terms from agt buffer, shape ntask,bs,dim
        obs_agt, act_agt, rew_agt, no_agt, terms_agt = self.sample_data(indices)
        sar_enc = self.prepare_encoder_data(obs_enc, act_enc, rew_enc) # squash o,a,r to one tensor
        # TODO check conflict between inc_enc and infer_freq # shape ntask*bs,dim
        v_pred_agt, pout_agt, task_z_agt = self.agent.forward(obs_agt, act_agt, no_agt, sar_enc.detach(), indices,
                                                                                              infer_freq=self.inc_enc if self.inc_enc>0 else self.infer_freq)
        # ntask,bs1+bs2
        cur_rew_pred = self.agent.pred_cur_rew(torch.cat((obs_agt,obs_enc),dim=1),
                                               torch.cat((act_agt,act_enc),dim=1)) # both agt and enc data shaped ntask,bs,dim
        ntask,bs,_ = cur_rew_pred.shape
        # TODO extend to infer freq
        rew_err, rew_loss = self.mse_crit(cur_rew_pred, torch.cat((rew_agt, rew_enc), dim=1), dim_start=2)
        rew_err = torch.mean(rew_err[:,rew_agt.size(1)],dim=1) # ntask,1

        new_act_agt, new_a_mean_agt, new_a_lstd_agt, new_a_logp_agt = pout_agt[:4] # 13,5,256,dim

        # the loss part
        if self.eq_enc: # when only decoder is used
            # the following is always used, even in sar2gam
            # from z to gamma p(gamma|z) < p(gamma|c,z)
            # i think this reconstruction is too sparse
            z = self.agent.sample_z(batch=z_bs)
            task_z_gam = self.agent.rec_gt_gamma(z)
            gammas_ = gammas.view(-1, 1, self.gamma_dim).repeat(z.size(0)//gammas.size(0), z_bs, 1)
            g_rec, rec_loss = self.mse_crit(task_z_gam, gammas_)
            if self.sar2gam and self.use_explorer: # from ci to gamma p(gamma|ci) < p(gamma|c,z)
                # as gamma and ci are observations, learning of ci to gamma is useless for the learning of z,
                # thus only used when explorer is enabled and serves as its reward
                task_gam = self.agent.ci2gam(sar_enc) # dim3?
                gammas = gammas.view(-1,1,self.gamma_dim)
                gammas_= gammas.repeat(1,task_gam.size(1),1)
                gam_rew = torch.sum((task_gam-gammas_)**2,dim=-1)
                # train with different batch of data
                data4enc = self.sample_data(indices, encoder=True, batchs=z_bs) # 20 sample for each task?
                # TODO is it necessary to use s,a with r
                data4enc = self.prepare_encoder_data(*data4enc[:3]) # s,a,r
                task_gam = self.agent.ci2gam(data4enc)
                gammas_ = gammas.repeat(1, task_gam.size(1), 1)
                g_rec2, rec_loss2 = self.mse_crit(gammas_, task_gam) # only used to train ci>gam
                self.writer.add_scalar('ci2gam_loss', rec_loss2, step)
                rec_loss += rec_loss2
                g_rec += g_rec2.repeat(g_rec.size(0)//g_rec2.size(0),1)

            # -kl(p(z|c)||p(z))+p(gamma|z,c)+p(c|z)
            kl_o, kl_div = self.agent.compute_kl_div(None) # pz with p(z|c), as q(z|c,gammma) is eq to p(z|c)
            # self.writer.add_scalar('kl', kl_div, step)
            kl_loss = self.kl_lambda * kl_div + self.rec_lambda * rec_loss
        elif self.use_ae: # using oencoder
            # z ~ q, p(gamma|z,c)

            gt_z = self.agent.infer_gt_z(gammas) # mb,dim > mb,zdim
            ## deterministic loss
            # task_z_gam = self.agent.rec_gt_gamma(gt_z)
            # g_rec, rec_loss = self.mse_crit(task_z_gam, gammas)
            ## stochastic
            gt_z_ = self.agent.sample_z(z_means=gt_z, batch=z_bs)
            task_z_gam = self.agent.rec_gt_gamma(gt_z_)
            gammas_ = gammas.view(-1, 1, self.gamma_dim).repeat(1, z_bs, 1)
            g_rec, rec_loss = self.mse_crit(task_z_gam, gammas_)
            # kl of p(z|gamma) and p(z|c)
            kl_o, kl_div = self.agent.compute_kl_div(gt_z)
            # if self.use_explorer:
            # even when explorer is not used, the kl should act as the extra constraint
            kl_o2, kl_div2 = self.agent.compute_kl_div(None) # TODO actually should use q(z|gamma)/p(z|c)logp(z)/p(z|c)
            self.writer.add_scalar('kl2', kl_div2, step)
            kl_div += kl_div2
            kl_o += kl_o2
            kl_loss = self.kl_lambda * kl_div + self.rec_lambda * rec_loss
        else: # the original pearl
            #### encourage the shortening of encoder data,
            # z_mean,z_var = self.agent.z_means, self.agent.z_vars # nup, ntask,1
            # tmp = [self.agent.compute_kl_div2((z_mean[i+1],z_var[i+1]),(z_mean[i],z_var[i])) for i in range(z_mean.size(0)-1)]
            # kl_o,kl_div = [torch.mean(torch.stack(x,dim=0),dim=0) for x in zip(*tmp)] # nup-1, ntask, 1; nup-1,1
            ##################
            kl_o,kl_div = self.agent.compute_kl_div() # kl_o:nup*ntask,1 mb,1
            g_rec = torch.zeros_like(kl_o)
            kl_loss = self.kl_lambda * kl_div

        self.writer.add_scalar('kl', kl_div, step)
        if self.use_ae or self.eq_enc: self.writer.add_scalar('g_rec', rec_loss, step) # gamma rec error

        if self.use_ae or self.eq_enc: self.dec_optimizer.zero_grad()
        if self.use_ae: self.enc_optimizer.zero_grad()
        if self.sar2gam: self.dec_optimizer2.zero_grad()
        # TODO consider gradients
        # encoder: context2z, kl loss, rew loss
        # rew_func: rew loss
        # value_net: soft value fit
        # actor: logp - q
        self.rew_optimizer.zero_grad()
        self.context_optimizer.zero_grad() # for task encoder
        (kl_loss+rew_loss).backward()
        self.context_optimizer.step()
        self.rew_optimizer.step()
        if self.eq_enc or self.use_ae: self.dec_optimizer.step()
        if self.use_ae:  self.enc_optimizer.step()
        if self.sar2gam: self.dec_optimizer2.step()

        if self.infer_freq!=0: # let the actor learn with multiple z
            obs_agt = obs_agt.unsqueeze(0).repeat(v_pred_agt.size(0), 1, 1, 1).view(-1, *obs_agt.shape[-2:])
            pout_agt = [x.view(-1,x.size(-1)) for x in pout_agt]
            v_pred_agt = v_pred_agt.view(-1,1)

        # optimize actor and valuenet
        new_a_q_agt,vloss_agt, agt_loss = self.optimize_p(self.vf_optimizer, self.agent,
                                                          (self.hpolicy_optimizer,self.lpolicy_optimizer) if self.dif_policy==1 else self.policy_optimizer,
                        obs_agt, v_pred_agt, pout_agt, terms_agt, dif_policy=self.dif_policy)

        self.writer.add_scalar('rew', rew_loss, step)
        self.writer.add_scalar('vf',vloss_agt, step)

        if self.use_explorer:
            ## TODO: combination, distribution
            if self.eq_enc and self.sar2gam:
                rew1 = -gam_rew
                if self.rew_mode==1: # combination reward mode
                    if self.infer_freq==0:
                        rew2 = -.01*rew_err - g_rec - kl_o # combination
                    else:
                        tmp =  -.01*rew_err.squeeze(-1) - g_rec.view(-1,num_tasks) - kl_o.view(-1,num_tasks) # nup, ntask
                        tmp = tmp*self.factors
                        tmp = tmp.repeat(self.infer_freq,1)[:self.batch_size] # bs,ntask,1
                        rew2 = tmp.transpose(1,0) # ntask,bs
                    rew_enc = rew1+rew2
                else: rew_enc = rew1
            else:
                # qloss_agt, rec_loss, kl div
                ## TODO if the density ratio is added to kl div2, this may also change
                if self.infer_freq==0:
                    rew_enc = -.01*rew_err - g_rec - kl_o  # e3,e1,e0, 2 kl term, 2 rec term, ## T
                    rew_enc = rew_enc.repeat(1, obs_enc.size(1))
                else:
                    # TODO add discounting factor
                    tmp = -.01 * rew_err.squeeze(-1) - g_rec.view(-1, num_tasks) - kl_o.view(-1,
                                                                                              num_tasks)  # nup, ntask
                    tmp = tmp.repeat(self.infer_freq, 1)[:self.batch_size]  # bs,ntask,1
                    rew_enc = tmp.transpose(1, 0).contiguous()  # ntask,bs

            # modified direct assignment of z
            mu, var = self.agent.z_means, self.agent.z_vars
            self.explorer.trans_z(mu[-num_tasks:], var[-num_tasks:])
            self.explorer.detach_z()
            # self.explorer.z = task_z[::self.batch_size]
            # i suppose this would be quite slow, as every iteration needs sampling
            if hasattr(rew_enc, 'detach'):
                rew_enc = rew_enc.detach()

            v_exp, pout_exp, _ = self.explorer.infer(obs_enc, act_enc, nobs_enc, task_z=None)
            exp_actions, exp_mean, exp_log_std, exp_log_pi = pout_exp[:4]

            exp_logp_target, vf_exp, exp_loss = self.optimize_p(self.vfexp_optimizer, self.explorer, self.exp_optimizer,
                                                                obs_enc, v_exp, pout_exp, terms_enc, dif_policy=0,
                                                                alpha=10)
            if step % 20 == 0:
                try:
                    self.writer.add_histogram('exp_adv', exp_logp_target - v_exp, step)
                    self.writer.add_histogram('logp_exp', exp_log_pi, step)
                    self.writer.add_histogram('a_logp', new_a_logp_agt, step)
                    if self.dif_policy==1: self.writer.add_histogram('e_logp', pout_agt[-1], step)
                    if self.sar2gam:
                        self.writer.add_histogram('trans_rew', rew1, step)
                        if self.rew_mode==1: self.writer.add_histogram('batch_rew',rew2, step)
                except Exception as e:
                    print(repr(e))
                    print('nan occurs')
            # self.writer.add_scalar('qf_exp', qf_exp, step)
            self.writer.add_scalar('vf_exp', vf_exp, step)
        #################################
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            # TODO this is kind of annoying and higher variance, why not just average
            # across all the train steps?
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['rew Loss'] = np.mean(ptu.get_numpy(rew_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vloss_agt))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                agt_loss
            ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Q Predictions',
            #     ptu.get_numpy(q1_pred_agt),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'V Predictions',
            #     ptu.get_numpy(v_pred_agt),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Log Pis',
            #     ptu.get_numpy(new_a_logp_agt),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Policy mu',
            #     ptu.get_numpy(new_a_mean_agt),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Policy log std',
            #     ptu.get_numpy(new_a_lstd_agt),
            # ))
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
        # agent.set_z(in_, idx)
        agent.infer_posterior(in_)
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
