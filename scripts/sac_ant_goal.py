"""
Run Prototypical Soft Actor Critic on HalfCheetahEnv.

"""
import numpy as np
import click
import datetime
import pathlib
import os
import sys
sys.path.append('/home/zhjl/oyster')
######################
import joblib
from scripts.shared import setup_nets
resume = False
exp_id = 'ant-goal'
exp_d = 'pearl-190421-114059'
resume_dir = os.path.join('output',f'{exp_id}',f'{exp_d}','params.pkl') # scripts/output/ant-goal/pearl-190417-112013
debug = True
use_explorer = True
use_ae = use_explorer and True
dif_policy = False
fast_debug = debug and True
exp_offp = False
confine_num_c = False
########################
from rlkit.envs.ant_goal import AntGoalEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.sac import ProtoSoftActorCritic
import rlkit.torch.pytorch_util as ptu

def datetimestamp(divider=''):
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f').replace('-', divider)

def experiment(variant, resume, note, debug, use_explorer, use_ae, dif_policy, test, confine_num_c, eq_enc, infer_freq):
    task_params = variant['task_params']
    env = NormalizedBoxEnv(AntGoalEnv(n_tasks=task_params['n_tasks'], use_low_gear_ratio=task_params['low_gear']))
    newenv = NormalizedBoxEnv(AntGoalEnv(n_tasks=task_params['n_tasks'], use_low_gear_ratio=task_params['low_gear']))
    ptu.set_gpu_mode(variant['use_gpu'], variant['gpu_id'])

    tasks = env.get_all_task_idx()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    physics_dim = 10
    z_dim = 5
    task_enc_output_dim = z_dim * 2 if variant['algo_params']['use_information_bottleneck'] else z_dim
    reward_dim = 1

    gamma_dim = 4 if use_ae else None # (a,r,rcosa,rsina)

    net_size = variant['net_size']
    # start with linear task encoding
    recurrent = variant['algo_params']['recurrent']

    memo = ''
    explorer = None
    if (resume or test) and resume_dir is not None:
        ret = joblib.load(resume_dir)
        agent_ = ret['exploration_policy']
        memo += f'this exp resumes {resume_dir}\n'
    # else:
    # share the task enc with these two agents
    agent, task_enc = setup_nets(recurrent, obs_dim, action_dim, reward_dim, task_enc_output_dim, net_size, z_dim, variant,
                                 dif_policy=dif_policy, task_enc=None, gt_ae=True if use_ae else None, gamma_dim=gamma_dim,
                                 confine_num_c=confine_num_c)
    explorer = setup_nets(recurrent, obs_dim, action_dim, reward_dim, task_enc_output_dim, net_size, z_dim, variant,
                          dif_policy=dif_policy, task_enc=task_enc, confine_num_c=confine_num_c)
    if resume or test:
        for snet,tnet in zip(agent_.networks,agent.networks):
            ptu.soft_update_from_to(snet, tnet, tau=1.)
    memo += f'[ant_goal] this exp wants to {note}'

    variant['algo_params']['memo'] = memo
    # modified train tasks eval tasks
    algorithm = ProtoSoftActorCritic(
        env=env,
        explorer=explorer if use_explorer else None,  # use the sequential encoder meaning using the new agent
        train_tasks=tasks[:] if debug else tasks[:-30],
        eval_tasks=tasks[:] if debug else tasks[-30:],
        agent=agent,
        latent_dim=z_dim,
        gamma_dim=gamma_dim,
        exp_offp=exp_offp,
        eq_enc=eq_enc,
        infer_freq=infer_freq,
        **variant['algo_params']
    )

    if ptu.gpu_enabled():
        algorithm.to()
    if test: algorithm.test(newenv)
    else: algorithm.train()


@click.command()
@click.argument('gpu', default=0)
@click.argument('debug', default=debug, type=bool)
@click.argument('use_explorer', default=use_explorer, type=bool)
@click.argument('use_ae',default=use_ae, type=bool)
@click.argument('dif_policy', default=dif_policy, type=bool)
@click.argument('exp_offp', default=exp_offp, type=bool)
@click.argument('confine_num_c', default=confine_num_c, type=bool) # make effect only when allowing extended exploration
@click.argument('eq_enc', default=False, type=bool)
@click.argument('infer_freq', default=0, type=int)
@click.option('--fast_debug', default=fast_debug, type=bool)
@click.option('--note', default='-')
@click.option('--resume', default=resume, is_flag=True) # 0 is false, any other is true
@click.option('--docker', default=0)
@click.option('--test', default=False, is_flag=True)
def main(gpu, debug, use_explorer, use_ae, dif_policy, exp_offp, confine_num_c, eq_enc, infer_freq,
         fast_debug, note, resume, docker, test):
    max_path_length = 200
    # noinspection PyTypeChecker
    # modified ntasks, meta-batch
    fast_debug = debug and fast_debug
    variant = dict(
        task_params=dict(
            n_tasks=8 if debug else 180, # 20 works pretty well
            randomize_tasks=True,
            low_gear=False,
        ),
        algo_params=dict(
            meta_batch=5 if debug else 8,
            num_iterations=10000,
            num_tasks_sample=5,
            num_steps_per_task=2 * max_path_length,
            num_train_steps_per_itr=4000 if not fast_debug else 1,
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
            replay_buffer_size=10000 if fast_debug else (10000 if debug else 1000000)
        ),
        cmd_params=dict(
            debug=debug,
            use_explorer = use_explorer,
            use_ae = use_explorer and use_ae,
            dif_policy = dif_policy,
            fast_debug = fast_debug,
            exp_offp = exp_offp,
            confine_num_c=confine_num_c,
            eq_enc=eq_enc,
            infer_freq=infer_freq,
        ),
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

    experiment(variant, resume, note, debug, use_explorer, use_ae, dif_policy, test, confine_num_c, eq_enc, infer_freq)

if __name__ == "__main__":
    main()
