import pybullet as pb
import numpy as np

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_block_pulling_planner import CloseLoopBlockPullingPlanner
from helping_hands_rl_envs.pybullet.utils.constants import NoValidPositionException

class CloseLoopBlockPullingEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)

  def reset(self):
    while True:
      self.resetPybulletWorkspace()
      self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
      try:
        if not self.random_orientation:
          padding = self._getDefaultBoarderPadding(constants.FLAT_BLOCK)
          min_distance = self._getDefaultMinDistance(constants.FLAT_BLOCK)
          x = np.random.random() * (self.workspace_size - padding) + self.workspace[0][0] + padding/2
          while True:
            y1 = np.random.random() * (self.workspace_size - padding) + self.workspace[1][0] + padding/2
            y2 = np.random.random() * (self.workspace_size - padding) + self.workspace[1][0] + padding/2
            if max(y1, y2) - min(y1, y2) > min_distance:
              break
          self._generateShapes(constants.FLAT_BLOCK, 2, pos=[[x, y1, self.object_init_z], [x, y2, self.object_init_z]], random_orientation=True)
        else:
          self._generateShapes(constants.FLAT_BLOCK, 2, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    return self.objects[0].isTouching(self.objects[1])

def createCloseLoopBlockPullingEnv(config):
  return CloseLoopBlockPullingEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.25, 0.65],
                          [-0.2, 0.2],
                          [0.01, 0.24]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'panda',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1, 1), 'hard_reset_freq': 1000}
  planner_config = {'random_orientation': False}
  env_config['seed'] = 1
  env = CloseLoopBlockPullingEnv(env_config)
  planner = CloseLoopBlockPullingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  done = False

  while not done:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)
