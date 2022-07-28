import pybullet as pb
import numpy as np

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_block_picking_planner import CloseLoopBlockPickingPlanner

class CloseLoopBlockReachingEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)

  def reset(self):
    self.resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
    return self._getObservation()

  def _checkTermination(self):
    gripper_pos = self.robot._getEndEffectorPosition()
    obj_pos = self.objects[0].getPosition()
    return np.linalg.norm(np.array(gripper_pos) - np.array(obj_pos)) < 0.03

def createCloseLoopBlockReachingEnv(config):
  return CloseLoopBlockReachingEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import numpy.random as npr
  workspace = np.asarray([[0.3, 0.6],
                          [-0.15, 0.15],
                          [0.01, 0.24]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': False, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': True, 'view_type': 'camera_center_xyz',
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'panda',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1, 1), 'hard_reset_freq': 1000}
  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi / 4}
  env_config['seed'] = npr.randint(10)
  env = CloseLoopBlockReachingEnv(env_config)
  planner = CloseLoopBlockPickingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  done = False

  while not done:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)
    plt.imshow(obs[2].squeeze(), cmap='gray'); plt.show()
