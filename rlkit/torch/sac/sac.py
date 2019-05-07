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
        self.dif_policy = self.agent.dif_policy

        self.reparameterize = reparameterize
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_ae = False
        if hasattr(self.agent, 'use_ae'):
            self.use_ae = self.agent.use_ae

        # TODO consolidate optimizers!
        if self.dif_policy:
            hpolicy_opt = optimizer_class(
                self.agent.hpolicy.parameters(),
                lr=policy_lr,
            )
            lpolicy_opt = optimizer_class(
                self.agent.lpolicy.parameters(),
                lr=policy_lr,
            )
            # recg_opt = optimizer_class(
            #     self.agent.recg.parameters(),
            #     lr=policy_lr,
            # )
            self.policy_optimizer = (hpolicy_opt,lpolicy_opt)
        else:
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
                lr=context_lr,
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

    def pretrain(self, n_iter=500, n_task=16):
        if self.use_explorer and self.eq_enc and self.sar2gam:
            for it in range(n_iter):
                indices = np.random.choice(self.train_tasks, size=n_task, replace=False)
                z_bs = 16
                gammas = self.make_variation(indices)
                data4enc = self.sample_data(indices, encoder=True, batchs=z_bs)  # 20 sample for each task?
                # TODO is it necessary to use s,a with r
                data4enc = self.prepare_encoder_data(*data4enc[:3])  # s,a,r
                task_gam = self.agent.ci2gam(data4enc)
                gammas_ = gammas.repeat(1, task_gam.size(1), 1)
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

    def optimize_q(self, qf1_optimizer, qf2_optimizer, rewards, num_tasks, terms, target_v_values, q1_pred, q2_pred, return_loss=True, scale_reward=True):
        # qf and encoder update (note encoder does not get grads from policy or vf)
        if return_loss:
            qf1_optimizer.zero_grad()
            qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(-1,1)
        # scale rewards for Bellman update
        if scale_reward: rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(-1,1)
        q_target = rewards_flat + (1 - terms_flat) * self.discount * target_v_values
        q1_err,q1_loss = self.mse_crit(q1_pred, q_target.detach())
        q2_err, q2_loss = self.mse_crit(q2_pred, q_target.detach())
        # err: mb.b
        q1_err,q2_err = [torch.mean(err.view(-1,self.batch_size),dim=-1,keepdim=True) for err in (q1_err,q2_err)]
        return q1_err+q2_err, q1_loss+q2_loss
        # error1 = (q1_pred - q_target.detach())
        # error2 = (q2_pred - q_target.detach())
        # if return_loss:
        #     # qpred qtarget=targetv
        #     qf_loss = torch.mean(error1**2) + torch.mean(error2**2)
        #     return error1,error2,qf_loss
        # else: return error1,error2
        # context_optimizer.step()

    def optimize_p(self, vf_optimizer, agent, policy_optimizer, obs, v_pred, a_out, dif_policy=False, alpha=1):
        # compute min Q on the new actions
        if dif_policy:
            eta_out,recg_logp,a_out = a_out
            eta,eta_mean,eta_logstd,eta_logp,eta_ptan = eta_out
            act, a_mean, a_logstd, a_logp, a_ptan = a_out
            a_logp += eta_logp+recg_logp # the new soft value: two entropy term, one cross entropy term
        else:
            act,a_logp,a_mean, a_logstd,a_ptan = a_out
        min_q_new_actions = agent.min_q(obs, act, agent.z.detach())

        ######### vf loss involves the gradients of
        # v
        ###########
        a_logp = a_logp*alpha
        # TODO: incorporte new soft q
        ## two entropy term : logp for eta and for action
        ## one cross entropy term: with recognition model
        ## the new agent should include three components:
        ## 1. s,z -> eta
        ## 2. s,eta -> a
        ## 3. s,a -> eta
        v_target = min_q_new_actions - a_logp
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        vf_optimizer.zero_grad()
        vf_loss.backward()
        # self.writer.add_scalar('vf', vf_loss, step)
        vf_optimizer.step()
        agent._update_target_network()

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
        # pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (a_ptan ** 2).sum(dim=1).mean()
        )
        if dif_policy:
            mean_reg_loss += self.policy_mean_reg_weight * (eta_mean ** 2).mean()
            std_reg_loss += self.policy_std_reg_weight * (eta_logstd ** 2).mean()

        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss
        if dif_policy:
            [opt.zero_grad() for opt in policy_optimizer]
        else: policy_optimizer.zero_grad()
        policy_loss.backward()
        if dif_policy:
            [opt.step() for opt in policy_optimizer]
        else:policy_optimizer.step()
        return log_policy_target, vf_loss, policy_loss

    def mse_crit(self, x, y):
        # make sure x,y has same dim
        nreduced = torch.mean((x-y)**2, dim=tuple(range(1,len(x.shape))), keepdim=True).view(-1,1) # mb,1
        reduced = torch.mean(nreduced)
        return nreduced,reduced


    def _take_step(self, indices, obs_enc, act_enc, rew_enc, nobs_enc, terms_enc, gammas=None):
        global step
        z_bs = 20
        num_tasks = len(indices)

        # prepare rl data and enc data
        obs_agt, act_agt, rew_agt, no_agt, terms_agt = self.sample_data(indices)
        sar_enc = self.prepare_encoder_data(obs_enc, act_enc, rew_enc)

        # get data through q-net v-net actor task-encoder
        q1_pred_agt, q2_pred_agt, v_pred_agt, aout_agent, target_v_agt, task_z_agt = self.agent(obs_agt, act_agt, no_agt, sar_enc.detach(), indices)
        if self.dif_policy: # only used by actor, not by explorer
            eta_agt,recg_logp, pout_agent = aout_agent
            eta_logp = eta_agt[-2]
        else: pout_agent = aout_agent
            # new_eta_agt, new_eta_mean_agt, new_eta_lstd_agt, new_eta_logp_agt, new_eta_ptan_agt = pout_agent
        new_act_agt, new_a_mean_agt, new_a_lstd_agt, new_a_logp_agt,new_a_ptan_agt = pout_agent

        if self.use_ae or self.eq_enc: self.dec_optimizer.zero_grad()
        if self.use_ae: self.enc_optimizer.zero_grad()
        if self.sar2gam: self.dec_optimizer2.zero_grad()
        self.context_optimizer.zero_grad() # for task encoder
        ## TODO let eq_enc and use_ae exlusive
        # the loss part
        if self.eq_enc: # when only decoder is used
            # the following is always used, even in sar2gam
            # from z to gamma p(gamma|z) < p(gamma|c,z)
            # i think this reconstruction is too sparse
            z = self.agent.sample_z(batch=z_bs)
            task_z_gam = self.agent.rec_gt_gamma(z)
            gammas_ = gammas.view(-1, 1, self.gamma_dim).repeat(1, z_bs, 1)
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
                g_rec += g_rec2

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
            kl_o,kl_div = self.agent.compute_kl_div() # mb,1
            g_rec = torch.zeros_like(kl_o)
            kl_loss = self.kl_lambda * kl_div

        self.writer.add_scalar('kl', kl_div, step)
        if self.use_ae or self.eq_enc: self.writer.add_scalar('g_rec', rec_loss, step) # gamma rec error
        ##########
        # qf loss involves gradients of
        # q1 q2 task encoder
        ##########
        # get loss for q-net
        qerr_agt,qloss_agt = self.optimize_q(self.qf1_optimizer, self.qf2_optimizer,
                        rew_agt, num_tasks, terms_agt, target_v_agt, q1_pred_agt, q2_pred_agt)
        # qerr_agt = qerr_agt.view(-1,self.batch_size)
        (kl_loss+qloss_agt).backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()
        if self.eq_enc or self.use_ae: self.dec_optimizer.step()
        if self.use_ae:  self.enc_optimizer.step()
        if self.sar2gam: self.dec_optimizer2.step()

        # TODO necessary to use explicit task_z ?
        new_a_q_agt,vloss_agt, agt_loss = self.optimize_p(self.vf_optimizer, self.agent, self.policy_optimizer,
                        obs_agt,  v_pred_agt, aout_agent, dif_policy=self.dif_policy) # aout can include two tuples
        # self.writer.add_histogram('act_adv', log_pi - log_pi_target + v_pred, step)
        # self.writer.add_histogram('logp',new_a_logp_agt, step)
        self.writer.add_scalar('qf', qloss_agt, step)
        self.writer.add_scalar('vf',vloss_agt, step)
        # put exp opt after q and before task enc

        if self.use_explorer:
            ## TODO: combination, distribution
            if self.eq_enc and self.sar2gam:
                rew_enc = -gam_rew
            else:
                # qloss_agt, rec_loss, kl div
                ## TODO if the density ratio is added to kl div2, this may also change
                rew_enc = -.01*qerr_agt - g_rec - kl_o  # e3,e1,e0, 2 kl term, 2 rec term, ## T

                rew_enc = rew_enc.repeat(1, obs_enc.size(1))

            # modified direct assignment of z
            mu, var = self.agent.z_means, self.agent.z_vars
            self.explorer.trans_z(mu, var)
            self.explorer.detach_z()
            # self.explorer.z = task_z[::self.batch_size]
            # i suppose this would be quite slow, as every iteration needs sampling
            if hasattr(rew_enc, 'detach'):
                rew_enc = rew_enc.detach()

            q1_exp, q2_exp, v_exp, pout_exp, target_v_exp, _ = self.explorer.infer(obs_enc, act_enc, nobs_enc, task_z=None)
            new_act_exp, new_a_mean_exp, new_a_lstd_exp, new_a_logp_exp, new_a_ptan_exp = pout_exp
            # exp_actions, exp_mean, exp_log_std, exp_log_pi,exp_tanh_value = pout_agent

            _, qfloss_exp = self.optimize_q(self.qf1exp_optimizer, self.qf2exp_optimizer,
                                           rew_enc, num_tasks, terms_enc, target_v_exp, q1_exp, q2_exp, scale_reward=True)
            self.qf1exp_optimizer.step()
            self.qf2exp_optimizer.step()
            exp_logp_target, vfloss_exp, exp_loss = self.optimize_p(self.vfexp_optimizer, self.explorer, self.exp_optimizer,
                                                                obs_enc, v_exp,
                                                                pout_exp,alpha=10)
            if step % 20 == 0:
                self.writer.add_histogram('exp_adv', exp_logp_target - v_exp, step)
                self.writer.add_histogram('logp_exp', new_a_logp_exp, step)
                self.writer.add_histogram('a_logp', new_a_logp_agt, step)
                self.writer.add_histogram('zm', self.agent.z_means, step)
                self.writer.add_histogram('q1pred', q1_pred_agt, step)
                if self.dif_policy:
                    self.writer.add_histogram('eta_logp',eta_logp, step)
                    self.writer.add_histogram('recg_logp',recg_logp, step)

            self.writer.add_scalar('qf_exp', qfloss_exp, step)
            self.writer.add_scalar('vf_exp', vfloss_exp, step)
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

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qloss_agt))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vloss_agt))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                agt_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred_agt),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred_agt),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(new_a_logp_agt),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(new_a_mean_agt),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(new_a_lstd_agt),
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
