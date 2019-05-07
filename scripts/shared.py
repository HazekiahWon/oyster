from rlkit.torch.sac.policies import TanhGaussianPolicy, EmbPolicy, HierPolicy, DecomposedPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.proto import ProtoAgent
from torch import nn
def setup_nets(recurrent, obs_dim, action_dim, reward_dim, task_enc_output_dim, net_size, z_dim, eta_dim, variant,
               dif_policy=False, obs_emb=False, task_enc=None, gt_ae=None, gamma_dim=10, confine_num_c=False, eq_enc=False,
               sar2gam=False):
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    is_actor = task_enc is None
    if task_enc is None:
        task_enc = encoder_model(
            hidden_sizes=[200, 200, 200],  # deeper net + higher dim space generalize better
            input_size=obs_dim + action_dim + reward_dim,
            output_size=task_enc_output_dim,
            # hidden_init=nn.init.xavier_normal_,
            # layer_norm=True
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
    if dif_policy:
        policy_cls = EmbPolicy if obs_emb else TanhGaussianPolicy
        hpolicy = policy_cls( # s,z
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim,
            lat_dim=z_dim,
            action_dim=eta_dim,
        )
        lpolicy = policy_cls(  # s,z
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim,
            lat_dim=eta_dim,
            action_dim=action_dim,
        )
        policy = HierPolicy(hpolicy, lpolicy)
        # policy = DecomposedPolicy(obs_dim,
        #                            z_dim=z_dim,
        #                            # latent_dim=64,
        #                            eta_nlayer=None,
        #                            num_expz=32,
        #                            atn_type='low-rank',
        #                            action_dim=action_dim,
        #                            anet_sizes=[net_size, net_size, net_size])
    else:
        # policy = TanhGaussianPolicy(
        #     hidden_sizes=[net_size, net_size, net_size],
        #     obs_dim=obs_dim,
        #     lat_dim=z_dim,
        #     action_dim=action_dim,
        # )
        policy = DecomposedPolicy(obs_dim,
                                  z_dim=z_dim,
                                  # latent_dim=64,
                                  eta_nlayer=None,
                                  num_expz=32,
                                  atn_type='low-rank',
                                  action_dim=action_dim,
                                  anet_sizes=[net_size, net_size, net_size])
    # policy2 = DecomposedPolicy(obs_dim,
    #                            z_dim=z_dim,
    #                            # latent_dim=64,
    #                            eta_nlayer=None,
    #                            num_expz=32,
    #                            atn_type='low-rank',
    #                            action_dim=action_dim,
    #                            anet_sizes=[net_size, net_size, net_size])

    nets = [task_enc, policy, qf1, qf2, vf]
    if is_actor:
        # gt_ae eq enc both
        # no gt ae, eq enc, dec
        # no both, none
        # no eq enc, gt_ae, both
        if gt_ae is not None :
            gt_encoder = encoder_model(
                hidden_sizes=[32, 32],  # deeper net + higher dim space generalize better
                input_size=gamma_dim,
                output_size=task_enc_output_dim//2,
                hidden_init=nn.init.xavier_normal_,
                layer_norm=True

            )
            nets = nets + [gt_encoder]

        if gt_ae is not None or eq_enc:
            # for walker: 6464; 64128
            gt_decoder = encoder_model(
                hidden_sizes=[64, 64],  # deeper net + higher dim space generalize better
                input_size=task_enc_output_dim // 2,
                output_size=gamma_dim,
                # output_activation=nn.Softmax(dim=-1), # predict as label
                hidden_init=nn.init.xavier_normal_,
                layer_norm=True
            )
            nets = nets + [gt_decoder]
            if sar2gam:
                gt_decoder2 = encoder_model(
                    hidden_sizes=[64, 128],  # deeper net + higher dim space generalize better
                    input_size=obs_dim+action_dim+reward_dim,
                    output_size=gamma_dim,
                    # output_activation=nn.Softmax(dim=-1), # predict as label
                    hidden_init=nn.init.xavier_normal_,
                    layer_norm=True
                )
                nets = nets + [gt_decoder2]



    agent = ProtoAgent(
        z_dim,
        nets,
        use_ae=gt_ae is not None,
        confine_num_c=confine_num_c,
        **variant['algo_params']
    )
    if is_actor: return agent, task_enc
    else: return agent
