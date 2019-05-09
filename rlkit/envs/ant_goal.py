import numpy as np
from rlkit.envs.ant_multitask_base import MultitaskAntEnv

# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
class AntGoalEnv(MultitaskAntEnv):
    def __init__(self, task={}, n_tasks=2, **kwargs):
        super(AntGoalEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self, num_tasks):
        v1 = np.random.random(num_tasks)
        a = v1 * 2 * np.pi
        v2 = np.random.random(num_tasks)
        r = 3 * v2 ** 0.5
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal, 'variation':np.array((v11, v22))} for goal,v11,v22 in zip(goals,v1,v2)]
        # tasks = [{'goal': goal, 'variation': np.array((ae,re,re*np.cos(ae),re*np.sin(ae)))} for goal, ae,re in zip(goals, a, r)]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
