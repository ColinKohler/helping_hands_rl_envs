import pybullet as pb
import numpy as np

from helping_hands_rl_envs.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.planners.close_loop_block_in_bowl_planner import CloseLoopBlockInBowlPlanner
from helping_hands_rl_envs.pybullet.utils.constants import NoValidPositionException

class CloseLoopBlockInBowlEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)
    self.bin_size = 0.25

  def initialize(self):
    super().initialize()

  def reset(self):
    while True:
      self.resetPybulletWorkspace()
      try:
        self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.BOWL, 1, scale=0.76, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    # check if bowl is upright
    if not self._checkObjUpright(self.objects[1]):
      return False
    # check if bowl and block is touching each other
    if not self.objects[0].isTouching(self.objects[1]):
      return False
    block_pos = self.objects[0].getPosition()[:2]
    bowl_pos = self.objects[1].getPosition()[:2]
    return np.linalg.norm(np.array(block_pos) - np.array(bowl_pos)) < 0.03

  def isSimValid(self):
    for obj in self.objects:
      p = obj.getPosition()
      if self._isObjectHeld(obj):
        continue
      if not self.workspace[0][0]-0.05 < p[0] < self.workspace[0][1]+0.05 and \
          self.workspace[1][0]-0.05 < p[1] < self.workspace[1][1]+0.05 and \
          self.workspace[2][0] < p[2] < self.workspace[2][1]:
        return False
    return True

def createCloseLoopBlockInBowlEnv(config):
  return CloseLoopBlockInBowlEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0.01, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'panda',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1, 1), 'hard_reset_freq': 1000}
  planner_config = {'random_orientation': False, 'dpos': 0.05, 'drot': np.pi/4}
  env_config['seed'] = 1
  env = CloseLoopBlockInBowlEnv(env_config)
  planner = CloseLoopBlockInBowlPlanner(env, planner_config)

  for _ in range(20):
    s, in_hand, obs = env.reset()
    done = False

    while not done:
      action = planner.getNextAction()
      obs, reward, done = env.step(action)
