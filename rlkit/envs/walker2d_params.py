from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv as Walker2dEnv

class Walker2dParamsEnv(Walker2dEnv):

    def __init__(self, task={}, n_tasks=2, sample_mode=0):
        super(Walker2dParamsEnv, self).__init__()
        self._task = task
        self.tasks = self.sample_tasks(n_tasks, sample_mode) #这里存放所有tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self.reset()
        self.set_task(self.tasks[idx])
        self._task = self.tasks[idx]
        self._goal = 0 # foo

