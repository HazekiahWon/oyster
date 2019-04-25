from rlkit.torch.sac.policies import TanhGaussianPolicy, DecomposedPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.proto import ProtoAgent
from torch import nn
def setup_nets(recurrent, obs_dim, action_dim, reward_dim, task_enc_output_dim, net_size, z_dim, variant, dif_policy=False, task_enc=None, gt_ae=None, gamma_dim=10):
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    is_actor = task_enc is None
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
                               # latent_dim=64,
                               eta_nlayer=None,
                               num_expz=32,
                               atn_type='low-rank',
                               action_dim=action_dim,
                               anet_sizes=[net_size, net_size, net_size])

    nets = [task_enc, policy2 if dif_policy else policy, qf1, qf2, vf]
    if is_actor and gt_ae is not None:
        gt_encoder = encoder_model(
            hidden_sizes=[64, 32],  # deeper net + higher dim space generalize better
            input_size=gamma_dim,
            output_size=task_enc_output_dim//2,
            hidden_init=nn.init.xavier_normal_,
            layer_norm=True

        )
        gt_decoder = encoder_model(
            hidden_sizes=[32, 64],  # deeper net + higher dim space generalize better
            input_size=task_enc_output_dim//2,
            output_size=gamma_dim,
            # output_activation=nn.Softmax(dim=-1), # predict as label
        hidden_init = nn.init.xavier_normal_,
            layer_norm=True
        )
        nets = nets + [gt_encoder, gt_decoder]

    agent = ProtoAgent(
        z_dim,
        nets,
        use_ae=gt_ae is not None,
        **variant['algo_params']
    )
    if is_actor: return agent, task_enc
    else: return agent
