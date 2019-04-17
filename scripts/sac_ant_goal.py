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
from rlkit.envs.ant_goal import AntGoalEnv

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy, DecomposedPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import ProtoSoftActorCritic
from rlkit.torch.sac.proto import ProtoAgent
import rlkit.torch.pytorch_util as ptu

def datetimestamp(divider=''):
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f').replace('-', divider)

def experiment(variant):
    task_params = variant['task_params']
    env = NormalizedBoxEnv(AntGoalEnv(n_tasks=task_params['n_tasks'], use_low_gear_ratio=task_params['low_gear']))
    ptu.set_gpu_mode(variant['use_gpu'], variant['gpu_id'])

    tasks = env.get_all_task_idx()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    physics_dim = 10
    z_dim = 5
    task_enc_output_dim = z_dim * 2 if variant['algo_params']['use_information_bottleneck'] else z_dim
    reward_dim = 1

    net_size = variant['net_size']
    # start with linear task encoding
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    task_enc = encoder_model(
            hidden_sizes=[200, 200, 200], # deeper net + higher dim space generalize better
            input_size=obs_dim + action_dim + reward_dim,
            output_size=task_enc_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + z_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + z_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + z_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + z_dim,
        # latent_dim=z_dim,
        action_dim=action_dim,
    )
    policy2 = DecomposedPolicy(obs_dim,
                               z_dim=z_dim,
                               latent_dim=64,
                               eta_nlayer=None,
                               num_expz=64,
                               action_dim=action_dim,
                               anet_sizes=[net_size, net_size, net_size])

    agent = ProtoAgent(
        z_dim,
        [task_enc, policy, qf1, qf2, vf],
        **variant['algo_params']
    )

    memo = 'this exp wants to reproduce pearl results in humanoid_dir'

    variant['algo_params']['memo'] = memo

    algorithm = ProtoSoftActorCritic(
        env=env,
        use_explorer=False,  # use the sequential encoder meaning using the new agent
        train_tasks=tasks[:-30],
        eval_tasks=tasks[-30:],
        agent=agent,
        latent_dim=z_dim,
        **variant['algo_params']
    )

    if ptu.gpu_enabled():
        algorithm.to()
    algorithm.train()


@click.command()
@click.argument('gpu', default=0)
@click.option('--docker', default=0)
def main(gpu, docker):
    max_path_length = 200
    # noinspection PyTypeChecker
    variant = dict(
        task_params=dict(
            n_tasks=180, # 20 works pretty well
            randomize_tasks=True,
            low_gear=False,
        ),
        algo_params=dict(
            meta_batch=10,
            num_iterations=10000,
            num_tasks_sample=5,
            num_steps_per_task=2 * max_path_length,
            num_train_steps_per_itr=4000,
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
        ),
        net_size=300,
        use_gpu=True,
        gpu_id=gpu,
    )
    exp_name = 'pearl'

    log_dir = '/mounts/output' if docker == 1 else 'output'
    exp_id = 'ant-goal'
    os.makedirs(os.path.join(log_dir, exp_id), exist_ok=True)
    experiment_log_dir = setup_logger(exp_name, variant=variant, exp_id=exp_id, base_log_dir=log_dir)

    # creates directories for pickle outputs of trajectories (point mass)
    pickle_dir = experiment_log_dir + '/eval_trajectories'
    pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)
    variant['algo_params']['output_dir'] = pickle_dir

    # debugging triggers a lot of printing
    DEBUG = 0
    os.environ['DEBUG'] = str(DEBUG)

    experiment(variant)

if __name__ == "__main__":
    main()
