import numpy as np
from rlkit.core import eval_util
def random_choice(tensor, size=256):
    if size>tensor.size(-2): return tensor
    else:
        indices = np.random.choice(np.arange(tensor.size(-2)), size, replace=False)
    return tensor[...,indices,:]

def rollout(env, agent,max_path_length=np.inf, animated=False, need_cupdate=True, infer_freq=0, deterministic=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    will only resample z and collect context, but will not update z posterior
    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :param is_online: if True, update the task embedding after each transition
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    video = list()

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        path_length += 1
        if need_cupdate:
            agent.update_context([o, a, r, next_o, d])
            if infer_freq>0 and path_length%infer_freq==0:
                # c = random_choice(agent.context)
                # c = agent.context
                agent.infer_posterior(agent.context, deterministic=deterministic)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)

        if d:
            break
        o = next_o
        if animated:
            tmp = env.render(mode='human')
            video.append(tmp)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        frames=np.stack(video) if len(video)!=0 else None
    )

def act_while_explore(env, agent, env2, actor, freq=20, num_avg_test=2, max_path_length=np.inf, animated=False, infer_freq=0):
    """
    rollout actor for 3 times while explorer exploring another 20 transitions

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :param is_online: if True, update the task embedding after each transition
    :return:
    """

    o = env.reset()
    path_length = 0
    if animated:
        env.render()
    ret_seq = list()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)

        agent.update_context([o, a, r, next_o, d])
        if (path_length+1)%freq==0:
            # c = random_choice(agent.context)
            # c = agent.context
            agent.infer_posterior(agent.context)
            actor.trans_z(agent.z_means, agent.z_vars)
            test_paths = list()
            for _ in range(num_avg_test):
                test_paths.append(rollout(env2, actor, max_path_length, animated, need_cupdate=False, infer_freq=infer_freq))
            ret = eval_util.get_average_returns(test_paths) # average multiple paths
            ret_seq.append(ret)

        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()

    return ret_seq


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
