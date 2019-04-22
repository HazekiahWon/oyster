from rlkit.torch.sac.policies import TanhGaussianPolicy, DecomposedPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.proto import ProtoAgent
def setup_nets(recurrent, obs_dim, action_dim, reward_dim, task_enc_output_dim, net_size, z_dim, variant, task_enc=None):
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    if task_enc is None:
        task_enc = encoder_model(
            hidden_sizes=[200, 200, 200],  # deeper net + higher dim space generalize better
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
    if task_enc is None: return agent
    else: return agent, task_enc
