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
            q_imp=False,
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
        self.mse_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()

        self.eval_statistics = None
        self.kl_lambda = kl_lambda
        self.rec_lambda = rec_lambda
        self.gam_rew_lambda = gam_rew_lambda
        self.eq_enc = eq_enc
        self.q_imp = q_imp
        self.sar2gam = sar2gam
        self.reparameterize = reparameterize
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_ae = False
        if hasattr(self.agent, 'use_ae'):
            self.use_ae = self.agent.use_ae

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

    def optimize_q(self, qf1_optimizer, qf2_optimizer, rewards, num_tasks, terms, target_v_values, q1_pred, q2_pred, return_loss=True):
        # qf and encoder update (note encoder does not get grads from policy or vf)
        if return_loss:
            qf1_optimizer.zero_grad()
            qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(-1,1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(-1,1)
        q_target = rewards_flat + (1 - terms_flat) * self.discount * target_v_values
        error1 = (q1_pred - q_target.detach())
        error2 = (q2_pred - q_target.detach())
        if return_loss:
            # qpred qtarget=targetv
            qf_loss = torch.mean(error1**2) + torch.mean(error2**2)
            return error1,error2,qf_loss
        else: return error1,error2
        # context_optimizer.step()

    def optimize_p(self, vf_optimizer, agent, policy_optimizer, obs, new_actions, log_pi, v_pred, policy_mean, policy_log_std, pre_tanh_value, alpha=1):
        # compute min Q on the new actions
        min_q_new_actions = agent.min_q(obs, new_actions, agent.z.detach())

        ######### vf loss involves the gradients of
        # v
        ###########
        log_pi = log_pi*alpha
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
        ##### policy loss involves the gradients of
        # actor, does not involve task encoder, q
        #############################
        if self.reparameterize:
            policy_loss = (
                    log_pi - log_policy_target  # to make it around 0
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

    def _take_step(self, indices, obs_enc, act_enc, rew_enc, nobs_enc, terms_enc, gammas=None):
        global step

        num_tasks = len(indices)

        # prepare rl data and enc data
        obs_agt, act_agt, rew_agt, no_agt, terms_agt = self.sample_data(indices)
        sar_enc = self.prepare_encoder_data(obs_enc, act_enc, rew_enc)

        # get data through q-net v-net actor task-encoder
        q1_pred_agt, q2_pred_agt, v_pred_agt, pout_agt, target_v_agt, task_z_agt = self.agent(obs_agt, act_agt, no_agt, sar_enc.detach(), indices)
        new_act_agt, new_a_mean_agt, new_a_lstd_agt, new_a_logp_agt = pout_agt[:4]
        new_a_ptan_agt = pout_agt[-1]

        self.context_optimizer.zero_grad() # for task encoder
        gt_z = None
        kl_loss = None
        if self.eq_enc:
            gam_rew = 0.
            self.dec_optimizer.zero_grad()
            if self.sar2gam:
                task_gam = self.agent.rec_gt_gamma(sar_enc) # dim3?
                gammas = gammas.view(-1,1,self.gamma_dim)
                gammas_= gammas.repeat(1,task_gam.size(1),1)
                gam_rew = (task_gam-gammas_)**2
                data4enc = self.sample_data(indices, encoder=True, batchs=20) # 20 sample for each task?
                task_gam = self.agent.rec_gt_gamma(data4enc)
                gammas_ = gammas.repeat(1, task_gam.size(1), 1)
                rec_loss = self.mse_criterion(gammas_, task_gam)
            else:
                #### rec of gamma: task_enc > z > decoder > rec_gama <mse> gt_gamma
                # affect decoder and task enc
                task_z_gam = self.agent.rec_gt_gamma(self.agent.z)
                rec_loss = self.mse_criterion(task_z_gam, gammas)
        elif self.use_ae: # and not use ae
            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()
            # ae with eq encoders

            #### gt_gamma > gt_enc > gt_z > samp_z > decoder > rec_gam <mse> gt_gamma
            # affect ae
            #### rew: task_enc > z > decoder > task_z_gam <se> gt_gamma
            # affect exploration p
            gt_z = self.agent.infer_gt_z(gammas)
            ########################
            dists = [torch.distributions.Normal(z, ptu.ones(self.agent.z_dim)) for z in gt_z]
            gt_z2 = torch.stack([dist.rsample() for dist in dists], dim=0)
            rec_gam = self.agent.rec_gt_gamma(gt_z2)
            task_z_gam = self.agent.rec_gt_gamma(self.agent.z)
            gam_rew = -torch.sum((task_z_gam-gammas)**2,dim=1, keepdim=True) # mb,1
            gam_rew = gam_rew.repeat(1,self.batch_size)
            gam_rew = gam_rew.view(-1, 1)
            #####################
            # rec_gam = torch.nn.Softmax(rec_gam,dim=1)
            self.writer.add_histogram('rec_gamma', rec_gam, step)
            self.writer.add_histogram('gt_z', gt_z[0], step)
            self.writer.add_histogram('task_z',self.agent.z_means[0], step)
            # rec_loss = self.onehot_criterion(rec_gam, torch.cuda.LongTensor(indices)) # because it is quite small if averaged
            rec_loss = self.mse_criterion(rec_gam, gammas)
        kl_loss = rec_loss*self.rec_lambda
        self.writer.add_scalar('ae_rec_loss', rec_loss, step)

        # distance between z: enc_data - z <> z
        # affect task enc and encoder
        kl_div = self.agent.compute_kl_div(gt_z)
        self.writer.add_scalar('vae_kl', kl_div, step)
        ###########
        # kl loss involve gradients of
        # 1. no ae: task encoder (be close to prior)
        # 2. ae: gt ae + task encoder
        ###########
        if kl_loss is None: kl_loss = self.kl_lambda * kl_div
        else: kl_loss += self.kl_lambda * kl_div

        ##########
        # qf loss involves gradients of
        # q1 q2 task encoder
        ##########
        # get loss for q-net
        q1err_agt,q2err_agt,qloss_agt = self.optimize_q(self.qf1_optimizer, self.qf2_optimizer,
                        rew_agt, num_tasks, terms_agt, target_v_agt, q1_pred_agt, q2_pred_agt)
        (kl_loss+qloss_agt).backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        self.context_optimizer.step()
        if self.eq_enc or self.use_ae:
            self.dec_optimizer.step()
        if self.use_ae:  self.enc_optimizer.step()
        # TODO necessary to use explicit task_z ?
        new_a_q_agt,vloss_agt, agt_loss = self.optimize_p(self.vf_optimizer, self.agent, self.policy_optimizer,
                        obs_agt, new_act_agt, new_a_logp_agt, v_pred_agt, new_a_mean_agt, new_a_lstd_agt, new_a_ptan_agt)
        # self.writer.add_histogram('act_adv', log_pi - log_pi_target + v_pred, step)
        self.writer.add_histogram('logp',new_a_logp_agt, step)
        self.writer.add_scalar('qf', qloss_agt, step)
        self.writer.add_scalar('vf',vloss_agt, step)
        # put exp opt after q and before task enc
        if self.use_explorer:
            # task_z = task_z.detach()
            # modified direct assignment of z
            mu, var = self.agent.z_means, self.agent.z_vars
            self.explorer.trans_z(mu, var)
            self.explorer.detach_z()
            # self.explorer.z = task_z[::self.batch_size]
            # i suppose this would be quite slow, as every iteration needs sampling
            if self.q_imp:
                old_rew = qloss_agt.detach()
                trans = list()
                for idx in range(len(indices)):
                    self.env.reset_task(idx)
                    self.task_idx = idx
                    self.explorer.trans_z(mu[idx].unsqueeze(0), var[idx].unsqueeze(0))
                    paths, _ = self.eval_sampler.obtain_samples3(self.explorer, deterministic=False, max_trajs=1,
                                                                 accum_context=False)
                    path = paths[0]
                    self.enc_replay_buffer.add_path(idx, path)
                    trans.append((path['observations'], path['actions'], path['rewards'], path['terminals'],
                                  path['next_observations']))
                o, a, r, exp_terms, no = [np.stack(x) for x in zip(*trans)]  # o,a,r,terms, shaped 3dim
                context = [o, a, r, None, None]
                o, a, r, no = [ptu.from_numpy(x) for x in (o, a, r, no)]
                exp_terms = ptu.from_numpy(exp_terms.astype(np.int32))
                self.agent.update_context(context)
                self.agent.infer_posterior(self.agent.context)
                # reevaluate on the new inferred z
                q1_pred_agt, q2_pred_agt, v_pred_agt, pout_agt, target_v_agt, _ = self.agent.infer(obs_agt, act_agt, no_agt)
                q1err_agt, q2err_agt = self.optimize_q(None, None, rew_agt, num_tasks, terms_agt,
                                                 target_v_agt, q1_pred_agt, q2_pred_agt, return_loss=False)
                new_rew = torch.mean(q1err_agt ** 2) + torch.mean(q2err_agt ** 2)
                imp = (old_rew - new_rew) / torch.sqrt(old_rew ** 2 + new_rew ** 2)
                imp = imp.view(1, 1, 1)
                rew1 = imp.repeat(o.size(0), o.size(1), 1)
                rew2 = -imp.repeat(obs_enc.size(0), obs_enc.size(1), 1)
                rew_enc = torch.cat((rew1, rew2), dim=1)
                exp_obs = torch.cat((o, obs_enc), dim=1)
                exp_a = torch.cat((a, act_enc), dim=1)
                exp_no = torch.cat((no, nobs_enc), dim=1)
                self.explorer.trans_z(mu, var)
                q1_exp, q2_exp, v_exp, pout_exp, target_v_exp, _ = self.explorer.infer(exp_obs, exp_a, exp_no,
                                                                                          task_z=None)
                terms_enc = torch.cat((exp_terms, terms_enc), dim=1)
                obs_enc, act_enc, nobs_enc = exp_obs, exp_a, exp_no
            else:
                q1_exp, q2_exp, v_exp, pout_exp, target_v_exp, _ = self.explorer.infer(obs_enc, act_enc, nobs_enc,
                                                                                          task_z=None)
                ###############
                # rew1 = -torch.abs(error1 + error2) * self.exp_error_scale / 2.
                rew2 = gam_rew # because in eq enc, this is set as the loss

                rew_enc = self.gam_rew_lambda * rew2 #- (1 - self.gam_rew_lambda) * rew1  # small as possible
                rew_enc = rew_enc.detach()
                # self.writer.add_scalar('q rec loss', torch.mean(rew1), step)
                ###############
            exp_actions, exp_mean, exp_log_std, exp_log_pi = pout_exp[:4]
            exp_tanh_value = pout_exp[-1]
            _, _, qf_exp = self.optimize_q(self.qf1exp_optimizer, self.qf2exp_optimizer,
                                           rew_enc, num_tasks, terms_enc, target_v_exp, q1_exp, q2_exp)
            self.qf1exp_optimizer.step()
            self.qf2exp_optimizer.step()
            exp_logp_target, vf_exp, exp_loss = self.optimize_p(self.vfexp_optimizer, self.explorer, self.exp_optimizer,
                                                                obs_enc, exp_actions, exp_log_pi, v_exp,
                                                                exp_mean, exp_log_std, exp_tanh_value, alpha=10)
            if step % 20 == 0:
                self.writer.add_histogram('exp_adv', exp_logp_target - v_exp, step)
                self.writer.add_histogram('logp_exp', exp_log_pi, step)
            self.writer.add_scalar('qf_exp', qf_exp, step)
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
