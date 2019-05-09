# from rand_param_envs.gym.core import Env
from gym.core import Env
# from rand_param_envs.gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco import MujocoEnv

import numpy as np


class MetaEnv(Env):
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        raise NotImplementedError
    '''
    def log_diagnostics(self, paths, prefix):
        """
        Logs env-specific diagnostic information

        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        pass
    '''

class RandomEnv(MetaEnv, MujocoEnv):
    """
    This class provides functionality for randomizing the physical parameters of a mujoco model
    The following parameters are changed:
        - body_mass
        - body_inertia
        - damping coeff at the joints
    """
    RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

    def __init__(self, log_scale_limit, file_name, *args, rand_params=RAND_PARAMS, **kwargs):
        MujocoEnv.__init__(self, file_name, 4)
        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        self.log_scale_limit = log_scale_limit            
        self.rand_params = rand_params
        self.save_parameters()

    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        param_sets = []

        for _ in range(n_tasks):
            # body mass -> one multiplier for all body parts

            new_params = {}
            var = list()

            if 'body_mass' in self.rand_params: # 8
                v = np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_mass.shape)
                body_mass_multiplyers = np.array(1.5) ** v
                new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers
                var.append(body_mass_multiplyers.reshape((-1,)))

            # body_inertia
            if 'body_inertia' in self.rand_params: # 8,3
                v = np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_inertia.shape)
                body_inertia_multiplyers = np.array(1.5) ** v
                new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']
                var.append(body_inertia_multiplyers.reshape((-1,)))

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params: # 9
                v = np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.dof_damping.shape)
                dof_damping_multipliers = np.array(1.3) ** v
                new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)
                var.append(dof_damping_multipliers.reshape((-1,)))

            # friction at the body components
            if 'geom_friction' in self.rand_params: # 8,3
                v = np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.geom_friction.shape)
                dof_damping_multipliers = np.array(1.5) ** v
                new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)
                var.append(dof_damping_multipliers.reshape((-1,)))

            new_params['variation'] = np.concatenate(var, axis=0)
            param_sets.append(new_params)

        return param_sets

    def set_task(self, task):
        for param, param_val in task.items():
            if param=='variation': continue
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            #print(param, param_val)
            setattr(self.model, param, param_val)
        self.cur_params = task

    def get_task(self):
        return self.cur_params

    def save_parameters(self):
        self.init_params = {}
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.model.body_mass

        # body_inertia
        if 'body_inertia' in self.rand_params:
            self.init_params['body_inertia'] = self.model.body_inertia

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = self.model.dof_damping

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = self.model.geom_friction
        self.cur_params = self.init_params
