import pybullet as pb
import numpy as np
import skimage.transform as sk_transform
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators.pybullet.utils.renderer import Renderer
from helping_hands_rl_envs.simulators.pybullet.utils.ortho_sensor import OrthographicSensor
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util


class CloseLoopEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.robot.home_positions = [-0.4446, 0.0837, -2.6123, 1.8883, -0.0457, -1.1810, 0.0699, 0., 0., 0., 0., 0., 0., 0., 0.]
    self.robot.home_positions_joint = self.robot.home_positions[:7]

    self.ws_size = max(self.workspace[0][1] - self.workspace[0][0], self.workspace[1][1] - self.workspace[1][0])
    cam_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 1]
    target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    cam_up_vector = [-1, 0, 0]
    self.sensor = OrthographicSensor(cam_pos, cam_up_vector, target_pos, self.ws_size, 0.1, 1)

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def step(self, action):
    motion_primative, x, y, z, rot = self._decodeAction(action)
    current_pos = self.robot._getEndEffectorPosition()
    current_rot = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())

    bTg = transformations.euler_matrix(0, 0, current_rot[-1])
    bTg[:3, 3] = current_pos
    gTt = np.eye(4)
    gTt[:3, 3] = [x, y, z]
    bTt = bTg.dot(gTt)
    pos = bTt[:3, 3]

    # pos = np.array(current_pos) + np.array([x, y, z])
    rot = np.array(current_rot) + np.array(rot)
    rot_q = pb.getQuaternionFromEuler(rot)
    pos[0] = np.clip(pos[0], self.workspace[0, 0], self.workspace[0, 1])
    pos[1] = np.clip(pos[1], self.workspace[1, 0], self.workspace[1, 1])
    pos[2] = np.clip(pos[2], self.workspace[2, 0], self.workspace[2, 1])
    self.robot.moveTo(pos, rot_q, dynamic=True)
    if motion_primative == constants.PICK_PRIMATIVE and not self.robot.gripper_closed:
      self.robot.closeGripper()
      self.wait(100)
    elif motion_primative == constants.PLACE_PRIMATIVE and self.robot.gripper_closed:
      self.robot.openGripper()
      self.wait(100)
    self.robot.holding_obj = self.robot.getPickedObj(self.objects)
    obs = self._getObservation(action)
    done = self._checkTermination()
    reward = 1.0 if done else 0.0

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    return obs, reward, done

  def _getObservation(self, action=None):
    ''''''
    self.heightmap = self._getHeightmap()
    return self._isHolding(), None, self.heightmap.reshape([1, self.heightmap_size, self.heightmap_size])


  def _getHeightmap(self):
    gripper_pos = self.robot._getEndEffectorPosition()
    gripper_rz = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())[2]
    target_pos = [gripper_pos[0], gripper_pos[1], 0]
    T = transformations.euler_matrix(0, 0, gripper_rz)
    # cam_up_vector = [-1, 0, 0]
    cam_up_vector = T.dot(np.array([-1, 0, 0, 1]))[:3]

    self.sensor.setCamMatrix(gripper_pos, cam_up_vector, target_pos)
    heightmap = self.sensor.getHeightmap(self.heightmap_size)
    depth = -heightmap + gripper_pos[2]
    return depth


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0, 0.25]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 6, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1, 1)}
  planner_config = {'random_orientation': True}
  env = CloseLoopEnv(env_config)
  s, obs = env.reset()

  fig, axs = plt.subplots(4, 5, figsize=(25, 20))
  for i in range(20):
    action = [-1, 0, 0, -0.01, 0]
    obs, reward, done = env.step(action)
    axs[i//5, i%5].imshow(obs[1][0], vmax=0.3)
  fig.show()

  while True:
    action = [-1, 0, 0, -0.01, 0]
    obs, reward, done = env.step(action)
    plt.imshow(obs[2][0], vmax=0.3)
    plt.colorbar()
    plt.show()
