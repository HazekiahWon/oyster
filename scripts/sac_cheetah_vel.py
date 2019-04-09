"""
Run Prototypical Soft Actor Critic on HalfCheetahEnv.

"""
import os
import numpy as np
import click
import datetime
import pathlib
from gym.envs.mujoco import HalfCheetahEnv
import sys
sys.path.append('/home/zhjl/oyster')
from rlkit.envs.half_cheetah_vel import HalfCheetahVelEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy, DecomposedPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, OracleEncoder, OracleEncoder2
from rlkit.torch.sac.sac import ProtoSoftActorCritic
from rlkit.torch.sac.proto import ProtoAgent
import rlkit.torch.pytorch_util as ptu

def datetimestamp(divider=''):
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f').replace('-', divider)

def experiment(variant):
    task_params = variant['task_params']
    env = NormalizedBoxEnv(HalfCheetahVelEnv(n_tasks=task_params['n_tasks']))
    ptu.set_gpu_mode(variant['use_gpu'], variant['gpu_id'])

    tasks = env.get_all_task_idx()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    physics_dim = 10
    latent_dim = 5
    task_enc_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1

    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    task_enc = encoder_model(
            hidden_sizes=[200, 200, 200], # deeper net + higher dim space generalize better
            input_size=obs_dim + action_dim + reward_dim,
            output_size=task_enc_output_dim,
    )
    ############################ type1 oracle
    oracle1 = OracleEncoder(
        hidden_sizes=[200,200,200],
        input_size=obs_dim + action_dim + reward_dim + physics_dim,
        output_size=task_enc_output_dim,
    )
    oracle_decoder1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=latent_dim,
        output_size=physics_dim,
    )
    ###############################
    ################### type2 oracle
    phy_encoder = encoder_model(
            hidden_sizes=[200, 200, 200], # deeper net + higher dim space generalize better
            input_size=physics_dim,
            output_size=latent_dim,
    )
    sar_encoder = encoder_model(
            hidden_sizes=[200, 200, 200], # deeper net + higher dim space generalize better
            input_size=obs_dim+action_dim+reward_dim,
            output_size=latent_dim,
    )
    oracle2 = OracleEncoder2(
        hidden_sizes=[200, 200, 200],
        input_size=latent_dim*2,
        output_size=task_enc_output_dim, # to append information bottleneck
        encoder1= sar_encoder,
        encoder2=phy_encoder
    )
    oracle_decoder2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=latent_dim,
        output_size=physics_dim,
    )
    ###################################
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    policy2 = DecomposedPolicy(obs_dim,
            z_dim=latent_dim,
            latent_dim=64,
            eta_nlayer=None,
            num_expz=64,
            action_dim=action_dim,
            anet_sizes=[net_size, net_size, net_size])

    agent = ProtoAgent(
        latent_dim,
        [task_enc, policy2, qf1, qf2, vf],
        **variant['algo_params']
    )

    memo = 'this exp wants to check out the attentional embedding between z and observation to produce eta'

    variant['algo_params']['memo'] = memo

    algorithm = ProtoSoftActorCritic(
        env=env,
        train_tasks=tasks[:-30],
        eval_tasks=tasks[-30:],
        nets=[agent, task_enc, policy2, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )
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
            n_tasks=130,
            randomize_tasks=True,
        ),
        algo_params=dict(
            meta_batch=10,
            num_iterations=500, # meta-train epochs
            num_tasks_sample=5,
            num_steps_per_task=2 * max_path_length,
            num_train_steps_per_itr=2000,
            num_evals=2, # number of evals with separate task encodings
            num_steps_per_eval=2 * max_path_length,
            batch_size=256, # to compute training grads from
            embedding_batch_size=100,
            embedding_mini_batch_size=100,
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
            kl_lambda=.1,
            use_information_bottleneck=True,
            train_embedding_source='online_exploration_trajectories',
            eval_embedding_source='online_exploration_trajectories',
            recurrent=False, # recurrent or averaging encoder
            dump_eval_paths=False,
            replay_buffer_size=100000,
        ),
        net_size=300,
        use_gpu=True,
        gpu_id=gpu,
    )
    exp_name = 'pearl'

    log_dir = '/mounts/output' if docker == 1 else 'output'
    exp_id = 'half-cheetah-vel'
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
