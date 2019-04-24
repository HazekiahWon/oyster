from rlkit.samplers.util import rollout, act_while_explore
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.core import logger, eval_util
import numpy as np

class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, max_samples, max_path_length):
        self.env = env
        # self.policy = policy

        self.max_path_length = max_path_length
        self.max_samples = max_samples
        assert (
            max_samples >= max_path_length,
            "Need max_samples >= max_path_length"
        )

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, agent, deterministic=False, num_samples=None, is_online=False, need_cupdate=True):
        """
        sample n_eval traj for task exploration
        the env should be reset by the outer function
        :param deterministic:
        :param num_samples:
        :param is_online:
        :return:
        """
        # self.env.reset_task(idx)
        policy = MakeDeterministic(agent) if deterministic else agent
        paths = []
        n_steps_total = 0
        max_samp = self.max_samples
        if num_samples is not None:
            max_samp = num_samples
        while n_steps_total + self.max_path_length < max_samp: # to leave out one more path
            path = rollout(
                self.env, policy, max_path_length=self.max_path_length, is_online=is_online, need_cupdate=need_cupdate)
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths

    def obtain_samples2(self, agent, explore=True, deterministic=False, is_online=False, need_cupdate=True):
        """
        sample n_eval traj for task exploration
        :param deterministic:
        :param num_samples:
        :param is_online:
        :return:
        """
        policy = MakeDeterministic(agent) if deterministic else agent
        paths = []
        if explore:
            num_roll = self.max_samples//self.max_path_length-1
        else: num_roll = 1
        # n_steps_total = 0
        # max_samp = self.max_samples
        # if num_samples is not None:
        #     max_samp = num_samples
        for i in range(num_roll): # to leave out one more path
            path = rollout(
                self.env, policy, max_path_length=self.max_path_length, is_online=is_online, need_cupdate=need_cupdate)
            paths.append(path)
            if explore:
                policy.infer_posterior(policy.context)
                # policy.sample_z() # only allow the explorer to guess the z
            # n_steps_total += len(path['observations'])
        return paths

    def obtain_test_samples(self, explorer, actor, newenv, max_explore, freq=20, num_test_avg=3, deterministic=False, is_online=False):
        """
        sample n_eval traj for task exploration
        :param deterministic:
        :param num_samples:
        :param is_online:
        :return:
        """
        actor = MakeDeterministic(actor) if deterministic else actor
        explorer = MakeDeterministic(explorer) if deterministic else explorer
        seq = list()
        explorer.clear_z()
        for i in range(max_explore): # to leave out one more path
            seq.extend(act_while_explore(self.env, explorer, newenv, actor, freq, num_test_avg,
                                         max_path_length=self.max_path_length, is_online=is_online))
        return np.asarray(seq)
