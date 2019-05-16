from rlkit.torch.sac.policies import *
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.proto import ProtoAgent
from torch import nn
def identity(x): return x

def setup_nets(recurrent, obs_dim, action_dim, reward_dim, task_enc_output_dim, net_size, variant, configs,
               dif_policy=False, obs_emb=False, task_enc=None, gt_ae=None, confine_num_c=False, eq_enc=False,
               sar2gam=False):
    keynames = ['z_dim','eta_dim', 'gamma_dim', 'gam2z', 'z2gam', 'ci2gam', 'obsemb_sizes','obs_emb_dim','etanet_sizes','anet_sizes', 'gam_act', 'sample_mode']
    z_dim,eta_dim,gamma_dim,gam2z,z2gam,ci2gam, obsemb_sizes, obs_emb_dim, etanet_sizes, anet_sizes, gam_act, sample_mode = [configs.get(k) for k in keynames]
    if not isinstance(gamma_dim, int):
        gamma_dim = gamma_dim[sample_mode]
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    is_actor = task_enc is None
    if task_enc is None:
        task_enc = encoder_model(# c2z
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
    if dif_policy==0:
        policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim,
            # lat_dim=z_dim,
            action_dim=action_dim,
        )
    elif dif_policy==1:
        policy_cls = EmbPolicy if obs_emb else TanhGaussianPolicy
        hpolicy = policy_cls( # s,z
            hidden_sizes=[64,64],
            obs_dim=obs_dim,
            lat_dim=z_dim,
            action_dim=eta_dim,
            hidden_init=nn.init.xavier_normal_,
            layer_norm=True,
        )
        lpolicy = policy_cls(  # s,z
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim,
            lat_dim=eta_dim,
            action_dim=action_dim,
            hidden_init=nn.init.xavier_normal_,
            layer_norm=True,
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
    elif dif_policy==2:
        policy = BNHierPolicy(obs_dim,
                              z_dim,
                              action_dim,
                              obsemb_sizes=obsemb_sizes,
                              obs_emb_dim=obs_emb_dim,
                              etanet_sizes=etanet_sizes,
                              anet_sizes=anet_sizes,
                              eta_dim=eta_dim)
    elif dif_policy==3:# 3
        policy = BNHierPolicy(obs_dim,
                              z_dim,
                              action_dim,
                              obsemb_sizes=obsemb_sizes,
                              obs_emb_dim=obs_emb_dim,
                              etanet_sizes=etanet_sizes,
                              anet_sizes=anet_sizes,
                              eta_dim=eta_dim,
                              sparse=True)
    elif dif_policy>=4:
        policy = MDNPolicy(obs_dim,
                              z_dim,
                              action_dim,
                              obsemb_sizes=obsemb_sizes,
                              obs_emb_dim=obs_emb_dim,
                              etanet_sizes=etanet_sizes,
                              anet_sizes=anet_sizes,
                              eta_dim=eta_dim,
                              n_component=dif_policy)
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
        if gt_ae is not None : # gamma2z
            gt_encoder = encoder_model(
                hidden_sizes=gam2z,  # deeper net + higher dim space generalize better
                input_size=gamma_dim,
                output_size=task_enc_output_dim//2,
                # hidden_init=nn.init.xavier_normal_,
                # layer_norm=True

            )
            nets = nets + [gt_encoder]

        if gt_ae is not None or eq_enc:
            # for walker: 6464; 64128
            gt_decoder = encoder_model(# z2gam
                hidden_sizes=z2gam,  # deeper net + higher dim space generalize better
                input_size=task_enc_output_dim // 2,
                output_size=gamma_dim,
                output_activation=torch.tanh if gam_act=='tanh' else identity,
                # output_activation=nn.Softmax(dim=-1), # predict as label
                # hidden_init=nn.init.xavier_normal_,
                # layer_norm=True
            )
            nets = nets + [gt_decoder]
            if sar2gam:
                gt_decoder2 = encoder_model(# ci2gam
                    hidden_sizes=ci2gam,  # deeper net + higher dim space generalize better
                    input_size=obs_dim+action_dim+reward_dim,
                    output_size=gamma_dim,
                    output_activation=torch.tanh if gam_act == 'tanh' else identity
                    # output_activation=nn.Softmax(dim=-1), # predict as label
                    # hidden_init=nn.init.xavier_normal_,
                    # layer_norm=True
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
