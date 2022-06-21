import pybullet as pb
import numpy as np

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_block_picking_corner_planner import CloseLoopBlockPickingCornerPlanner
from helping_hands_rl_envs.pybullet.equipments.corner import Corner

class CloseLoopBlockPickingCornerEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)
    self.corner = Corner()
    self.corner_rz = 0
    self.corner_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]

  def resetCorner(self):
    self.corner_rz = np.random.random_sample() * 2*np.pi - np.pi if self.random_orientation else 0
    self.corner_pos = self._getValidPositions(0.22, 0, [], 1)[0]
    self.corner_pos.append(0)
    self.corner.reset(self.corner_pos, pb.getQuaternionFromEuler((0, 0, self.corner_rz)))

  def initialize(self):
    super().initialize()
    self.corner.initialize(pos=self.corner_pos)

  def reset(self):
    self.resetPybulletWorkspace()
    self.resetCorner()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    pos, rot_q = self.corner.getObjPose()

    self.cube = self._generateShapes(constants.CUBE, 1, pos=[pos], rot=[rot_q])[0]
    pb.changeDynamics(self.cube.object_id, -1, lateralFriction=0.75, mass=0.2)

    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    gripper_z = self.robot._getEndEffectorPosition()[-1]
    return self.robot.holding_obj == self.objects[-1] and gripper_z > 0.10

  def getObjectPoses(self, objects=None):
    if objects is None: objects = self.objects + [self.corner]

    obj_poses = list()
    for obj in objects:
      pos, rot = obj.getPose()
      rot = self.convertQuaternionToEuler(rot)

      obj_poses.append(pos + rot)
    return np.array(obj_poses)

def createCloseLoopBlockPickingCornerEnv(config):
  return CloseLoopBlockPickingCornerEnv(config)

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
  env = CloseLoopBlockPickingCornerEnv(env_config)
  planner = CloseLoopBlockPickingCornerPlanner(env, planner_config)

  num_success = 0
  for _ in range(20):
    obs = env.reset()
    done = False
    while not done:
      action = planner.getNextAction()
      plt.imshow(obs[2].squeeze(), cmap='gray'); plt.show()

      obs, reward, done = env.step(action)

    if reward > 0.9:
      num_success += 1
  print(num_success)
