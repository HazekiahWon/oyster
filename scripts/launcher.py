import numpy as np
import click
import datetime
import pathlib
import os
import sys
import json
sys.path.append('/home/zhjl/oyster')
######################
import joblib
from scripts.shared import setup_nets
resume = False
# exp_id = 'half-cheetah-vel'
# exp_d = 'pearl-190501-223401'
# resume_dir = os.path.join('output',f'{exp_id}',f'{exp_d}','params.pkl') # scripts/output/ant-goal/pearl-190417-112013
# debug = True
# use_explorer = True
# use_ae = use_explorer and True
# dif_policy = False
fast_debug = True
# exp_offp = False
# confine_num_c = False
########################
from rlkit.envs.half_cheetah_vel import HalfCheetahVelEnv
from rlkit.envs.ant_goal import AntGoalEnv
from rlkit.envs.walker2d_params import Walker2dParamsEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.sac import ProtoSoftActorCritic
import rlkit.torch.pytorch_util as ptu
env_cls = {
    "cheetah-vel": HalfCheetahVelEnv, # a:8, o:113
    "ant-goal": AntGoalEnv, # a:6, o:20
    "walker-params": Walker2dParamsEnv
}

def datetimestamp(divider=''):
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f').replace('-', divider)

def experiment(variant, resume, note, debug, use_explorer, use_ae, dif_policy, obs_emb, test, confine_num_c, eq_enc, infer_freq,
               rew_mode, sar2gam, exp_offp,
               configs):
    keynames = ['exp_id', 'resume_dir', 'num_eval_tasks', 'gamma_dim', 'z_dim', 'eta_dim','sample_mode','debug']
    exp_id, resume_dir, num_eval_tasks, gamma_dim, z_dim, eta_dim, sample_mode, debug = [configs.get(k) for k in keynames]
    Env = env_cls[exp_id]
    task_params = variant['task_params']
    env = NormalizedBoxEnv(Env(n_tasks=task_params['n_tasks'], sample_mode=sample_mode))
    # newenv = NormalizedBoxEnv(Env(n_tasks=task_params['n_tasks']))
    ptu.set_gpu_mode(variant['use_gpu'], variant['gpu_id'])

    tasks = env.get_all_task_idx()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    task_enc_output_dim = z_dim * 2 if variant['algo_params']['use_information_bottleneck'] else z_dim
    reward_dim = 1

    gamma_dim = gamma_dim if use_ae or eq_enc else None # only velocity
    if gamma_dim is not None and not isinstance(gamma_dim, int):
        gamma_dim = gamma_dim[sample_mode]

    net_size = variant['net_size']
    # start with linear task encoding
    recurrent = variant['algo_params']['recurrent']

    memo = ''
    explorer = None
    if (resume or test) and resume_dir is not None:
        ret = joblib.load(resume_dir)
        agent_ = ret['actor'] # the old version : exploration policy
        memo += f'this exp resumes {resume_dir}\n'
    # else:
    # share the task enc with these two agents
    agent, task_enc = setup_nets(recurrent, obs_dim, action_dim, reward_dim, task_enc_output_dim, net_size, variant, configs,
                                 dif_policy=dif_policy, obs_emb=obs_emb, task_enc=None, gt_ae=True if use_ae else None,
                                 confine_num_c=confine_num_c, eq_enc=eq_enc, sar2gam=sar2gam)
    explorer = setup_nets(recurrent, obs_dim, action_dim, reward_dim, task_enc_output_dim, net_size, variant, configs,
                          dif_policy=dif_policy, obs_emb=obs_emb, task_enc=task_enc, confine_num_c=confine_num_c)
    if resume or test:
        for snet,tnet in zip(agent_.networks,agent.networks):
            ptu.soft_update_from_to(snet, tnet, tau=1.)
    memo += f'[{exp_id}] this exp wants to {note}'

    variant['algo_params']['memo'] = memo
    # modified train tasks eval tasks
    algorithm = ProtoSoftActorCritic(
        env=env,
        explorer=explorer if use_explorer else None,  # use the sequential encoder meaning using the new agent
        train_tasks=tasks[:-2] if debug else tasks[:-num_eval_tasks],
        eval_tasks=tasks[-2:] if debug else tasks[-num_eval_tasks:],
        agent=agent,
        latent_dim=z_dim,
        gamma_dim=gamma_dim,
        exp_offp=exp_offp,
        eq_enc=eq_enc,
        infer_freq=infer_freq,
        rew_mode=rew_mode,
        sar2gam=sar2gam,
        dif_policy=dif_policy,
        **variant['algo_params']
    )

    if ptu.gpu_enabled():
        algorithm.to()
    # if test: algorithm.test(newenv)
    # else:
    algorithm.train(fast_debug=debug and fast_debug)

@click.command()
@click.argument('config', default=None, type=str)
@click.argument('gpu', default=0)
@click.argument('debug', default=False, type=bool)
@click.argument('use_explorer', default=False, type=bool)
@click.argument('use_ae',default=False, type=bool)
@click.argument('eq_enc', default=False, type=bool) # higher priority over ae
@click.argument('sar2gam', default=False, type=bool)
@click.argument('rew_mode', default=0, type=int)
@click.argument('dif_policy', default=0, type=int)
@click.argument('obs_emb', default=False, type=bool)
@click.argument('exp_offp', default=False, type=bool)
@click.argument('confine_num_c', default=False, type=bool) # make effect only when allowing extended exploration
@click.argument('infer_freq', default=0, type=int)
@click.argument('num_exp', default=1, type=int)
@click.option('--fast_debug', default=fast_debug, type=bool)
@click.option('--note', default='-')
@click.option('--resume', default=resume, is_flag=True) # 0 is false, any other is true
@click.option('--docker', default=0)
@click.option('--test', default=False, is_flag=True)
def main(config, gpu, debug, use_explorer, use_ae, dif_policy, obs_emb, exp_offp, confine_num_c, eq_enc, infer_freq, rew_mode, sar2gam, num_exp,
         fast_debug, note, resume, docker, test):
    configs = dict() # use a default json
    if config:
        with open(os.path.join(config+'.json')) as f:
            configs = json.load(f)
    keynames = ['exp_id', 'resume_dir', 'num_tasks', 'num_eval_tasks', 'gamma_dim', 'z_dim', 'eta_dim', 'num_iters']
    exp_id, resume_dir, num_tasks, num_eval_tasks, gamma_dim, z_dim, eta_dim, num_iters = [configs.get(k) for k in keynames]
    max_path_length = 200
    # noinspection PyTypeChecker
    # modified ntasks, meta-batch
    fast_debug = debug and fast_debug
    variant = dict(
        task_params=dict(
            n_tasks=8 if debug else num_tasks, # 20 works pretty well
            randomize_tasks=True,
            low_gear=False,
        ),
        algo_params=dict(
            meta_batch=5 if debug else 8,
            num_iterations=num_iters,
            num_tasks_sample=5,
            num_steps_per_task=2 * max_path_length,
            num_train_steps_per_itr=4000 if not fast_debug else 1,
            num_exp_traj_eval=num_exp,
            num_evals=2,
            num_steps_per_eval=2 * max_path_length,  # num transitions to eval on
            embedding_batch_size=256,
            embedding_mini_batch_size=256,
            batch_size=256, # to compute training grads from
            max_path_length=max_path_length,
            discount=0.99,
            soft_target_tau=0.005,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3e-4,
            reward_scale=5.,
            sparse_rewards=False,
            reparameterize=True,
            kl_lambda=1.,
            use_information_bottleneck=True,  # only supports False for now
            eval_embedding_source='online_exploration_trajectories',
            train_embedding_source='online_exploration_trajectories',
            recurrent=False, # recurrent or averaging encoder
            dump_eval_paths=False,
            replay_buffer_size=10000 if fast_debug else (10000 if debug else 100000) # buffer modified
        ),
        cmd_params=dict(
            debug=debug,
            use_explorer = use_explorer,
            use_ae = use_ae,
            dif_policy = dif_policy,
            fast_debug = fast_debug,
            exp_offp = exp_offp,
            confine_num_c=confine_num_c,
            eq_enc=eq_enc,
            infer_freq=infer_freq,
            rew_mode=rew_mode,
            num_exp=num_exp,
            sar2gam=sar2gam and use_explorer and eq_enc,  # only when explorer and eq enc are enabled, gives reward to explorer, cannot connect with encoder
        ),
        configs=configs,
        net_size=300,
        use_gpu=True,
        gpu_id=gpu,
    )
    exp_name = 'pearl'

    log_dir = '/mounts/output' if docker == 1 else 'output'

    os.makedirs(os.path.join(log_dir, exp_id), exist_ok=True)
    experiment_log_dir = setup_logger(exp_name, variant=variant, exp_id=exp_id, base_log_dir=log_dir)

    # creates directories for pickle outputs of trajectories (point mass)
    pickle_dir = experiment_log_dir + '/eval_trajectories'
    pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)
    variant['algo_params']['output_dir'] = pickle_dir

    # debugging triggers a lot of printing
    DEBUG = 0
    os.environ['DEBUG'] = str(DEBUG)

    experiment(variant, resume, note, debug, use_explorer, use_ae, dif_policy, obs_emb, test, confine_num_c, eq_enc, infer_freq,
               rew_mode, sar2gam, exp_offp,
               configs)

if __name__ == "__main__":
    main()
