import numpy as np
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_

class HalfCheetahEnv(HalfCheetahEnv_):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def render(self, mode='human'):
        if mode == 'rgb_array':
            # print('1')
            self._get_viewer(mode).render()
            # print('2')
            # window size used for old mujoco-py:
            width, height = 500, 500
            viewer = self._get_viewer(mode)
            data = viewer.read_pixels(width, height, depth=False)
            return data
        elif mode == 'human':
            viewer = self._get_viewer(mode)
            viewer.render()
            data = viewer._read_pixels_as_in_window()
            return data


