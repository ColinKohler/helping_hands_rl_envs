import pybullet as pb
import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.pybullet.utils import transformations
import helping_hands_rl_envs.pybullet.utils.object_generation as pb_obj_generation
from helping_hands_rl_envs.pybullet.equipments.square_peg_hole import SquarePegHole
from helping_hands_rl_envs.planners.close_loop_peg_insertion_planner import CloseLoopPegInsertionPlanner

class CloseLoopPegInsertionEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)
    self.peg_hole = SquarePegHole()
    self.peg_hole_rz = 0
    self.peg_hole_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]

  def resetPegHole(self):
    self.peg_hole_rz = np.random.random_sample() * 2*np.pi - np.pi if self.random_orientation else 0
    self.peg_hole_pos = self._getValidPositions(0.2, 0, [], 1)[0]
    self.peg_hole_pos.append(0)
    self.peg_hole.reset(self.peg_hole_pos, pb.getQuaternionFromEuler((0, 0, self.peg_hole_rz)))

  def initialize(self):
    super().initialize()
    self.peg_hole.initialize(pos=self.peg_hole_pos, rot=pb.getQuaternionFromEuler((0, 0, self.peg_hole_rz)))

  def reset(self):
    self.resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))

    self.resetPegHole()
    self.peg = self._generateShapes(constants.SQUARE_PEG, pos=[[self.workspace[0].mean(), self.workspace[1].mean(), 0.17]], rot=[[0,0,0,1]], scale=0.125, wait=False)[0]
    self.robot.closeGripper()
    self.setRobotHoldingObj()
    self.peg.resetPose([self.workspace[0].mean(), self.workspace[1].mean(), 0.17], [0,0,0,1])

    return self._getObservation()

  def _checkTermination(self):
    if not self._isHolding():
      return True

    hole_pos, hole_rot = self.peg_hole.getHolePose()
    peg_pos = self.peg.getPosition()

    return np.allclose(hole_pos[:2], peg_pos[:2], atol=1e-2) and peg_pos[2] < 0.1

  def _getReward(self):
    hole_pos, hole_rot = self.peg_hole.getHolePose()
    peg_pos = self.peg.getPosition()

    if np.allclose(hole_pos[:2], peg_pos[:2], atol=1e-2) and peg_pos[2] < 0.1:
      return 1
    else:
      return 0

def createCloseLoopPegInsertionEnv(config):
  return CloseLoopPegInsertionEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import time
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0.01, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'panda',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1, 1), 'hard_reset_freq': 1000,
                'view_type': 'camera_center_xyz'}
  planner_config = {'random_orientation': False, 'dpos': 0.025, 'drot': np.pi/8}
  env_config['seed'] = 1
  env = CloseLoopPegInsertionEnv(env_config)
  planner = CloseLoopPegInsertionPlanner(env, planner_config)

  input('start')
  for _ in range(20):
    obs = env.reset()
    done = False
    while not done:
      action = planner.getNextAction()
      finger_a_force, finger_b_force = env.robot.getFingerForce()
      finger_force = np.array([finger_a_force[:3], finger_b_force[:3]]).reshape(-1)

      obs, reward, done = env.step(action)
    print(reward)
