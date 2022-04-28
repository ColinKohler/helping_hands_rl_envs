import pybullet as pb
import numpy as np
from scipy.ndimage import rotate

from helping_hands_rl_envs.pybullet.utils import transformations
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_picking_corner import CloseLoopBlockPickingCornerEnv
from helping_hands_rl_envs.planners.close_loop_block_picking_corner_planner import CloseLoopBlockPickingCornerPlanner

class ForceBlockPickingCornerEnv(CloseLoopBlockPickingCornerEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getForceMap(self):
    finger_a_force, finger_b_force = self.robot.getFingerForce()
    finger_a_force_mag = np.sqrt(np.sum(finger_a_force ** 2))
    finger_b_force_mag = np.sqrt(np.sum(finger_b_force ** 2))
    #finger_a_force_mag, finger_b_force_mag = self.robot.getFingerForce()

    gripper_state = self.robot.getGripperOpenRatio()
    gripper_rz = pb.getEulerFromQuaternion(self.robot._getEndEffectorRotation())[2]
    gripper_max_open = 36 * self.workspace_size / self.obs_size_m # Panda specific

    force_map = np.zeros((self.heightmap_size, self.heightmap_size))
    finger_half_size = 5 * self.workspace_size / self.obs_size_m
    finger_half_size = round(finger_half_size / 128 * self.heightmap_size)

    d = int(gripper_max_open / 128 * self.heightmap_size * gripper_state)
    anchor = self.heightmap_size // 2
    force_map[int(anchor - d // 2 - finger_half_size):int(anchor - d // 2 + finger_half_size), int(anchor - finger_half_size):int(anchor + finger_half_size)] = finger_b_force_mag
    force_map[int(anchor + d // 2 - finger_half_size):int(anchor + d // 2 + finger_half_size), int(anchor - finger_half_size):int(anchor + finger_half_size)] = finger_a_force_mag
    force_map = rotate(force_map, np.rad2deg(gripper_rz), reshape=False, order=0)

    return force_map

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)

    #finger_a_force, finger_b_force = self.robot.getFingerForce()
    #finger_force = [np.sqrt(np.sum(finger_a_force[:3])**2), np.sqrt(np.sum(finger_b_force[:3])**2)]
    #finger_force = [finger_a_force[:3], finger_b_force[:3]]
    force_map = self._getForceMap()

    return state, hand_obs, obs, force_map

def createForceBlockPickingCornerEnv(config):
  return ForceBlockPickingCornerEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.25, 0.65],
                          [-0.2, 0.2],
                          [0.01, 0.25]])

  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': None, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'panda',
                'object_init_space_check': 'point', 'physics_mode': 'force', 'object_scale_range': (1.2, 1.2),
                'hard_reset_freq': 1000, 'view_type': 'camera_center_xyz'}
  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi/8}
  env = ForceBlockPickingCornerEnv(env_config)
  planner = CloseLoopBlockPickingCornerPlanner(env, planner_config)

  num_success = 0
  for _ in range(20):
    obs = env.reset()
    done = False
    t = 0
    while not done:
      action = planner.getNextAction()
      if t < 100:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(obs[2].squeeze(), cmap='gray')
        ax[1].imshow(obs[3].squeeze(), cmap='gray')
        plt.show()

      #print(np.round(finger_force, 2))

      obs, reward, done = env.step(action)
      t += 1

    if reward > 0.9:
      num_success += 1
  print(num_success)
