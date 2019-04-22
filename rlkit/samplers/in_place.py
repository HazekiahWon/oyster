from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic


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
    def __init__(self, env, policy, max_samples, max_path_length):
        self.env = env
        self.policy = policy

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

    def obtain_samples(self, deterministic=False, num_samples=None, is_online=False):
        """
        sample n_eval traj for task exploration
        :param deterministic:
        :param num_samples:
        :param is_online:
        :return:
        """
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        max_samp = self.max_samples
        if num_samples is not None:
            max_samp = num_samples
        while n_steps_total + self.max_path_length < max_samp: # to leave out one more path
            path = rollout(
                self.env, policy, max_path_length=self.max_path_length, is_online=is_online)
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths

    def obtain_samples2(self, explore=True, deterministic=False, is_online=False):
        """
        sample n_eval traj for task exploration
        :param deterministic:
        :param num_samples:
        :param is_online:
        :return:
        """
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
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
                self.env, policy, max_path_length=self.max_path_length, is_online=is_online)
            paths.append(path)
            # n_steps_total += len(path['observations'])
        return paths
