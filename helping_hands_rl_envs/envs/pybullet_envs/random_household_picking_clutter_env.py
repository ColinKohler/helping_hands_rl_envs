import math
from copy import deepcopy
import numpy as np
import os
import glob
import helping_hands_rl_envs
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.simulators.pybullet.equipments.tray import Tray
from scipy.ndimage.interpolation import rotate
import pybullet as pb
import os
import pybullet_data

def creat_duck(pos):
    shift = [0, -0.02, 0]
    scale = 0.08
    meshScale = [scale, scale, scale]
    #the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
    visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                        fileName="duck.obj",
                                        rgbaColor=[1, 1, 1, 1],
                                        specularColor=[0.4, .4, 0],
                                        visualFramePosition=shift,
                                        meshScale=meshScale)
    collisionShapeId = pb.createCollisionShape(shapeType=pb.GEOM_MESH,
                                              fileName="duck_vhacd.obj",
                                              collisionFramePosition=shift,
                                              meshScale=meshScale)

    pb.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=pos,
                      useMaximalCoordinates=True)


class RandomHouseholdPickingClutterEnv(PyBulletEnv):
  '''
  '''
  def __init__(self, config):
    super(RandomHouseholdPickingClutterEnv, self).__init__(config)
    self.object_init_z = 0.1
    self.obj_grasped = 0
    self.tray = Tray()
    self.exhibit_env_obj = False

  def initialize(self):
    super().initialize()
    self.tray.initialize(pos=[self.workspace[0].mean(), self.workspace[1].mean(), 0],
                         size=[self.workspace_size+0.015, self.workspace_size+0.015, 0.1])

  def _decodeAction(self, action):
    """
    decode input action base on self.action_sequence
    Args:
      action: action tensor

    Returns: motion_primative, x, y, z, rot

    """
    primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a), ['p', 'x', 'y', 'z', 'r'])
    motion_primative = action[primative_idx] if primative_idx != -1 else 0
    if self.action_sequence.count('r') <= 1:
      rz = action[rot_idx] if rot_idx != -1 else 0
      ry = 0
      rx = 0
    else:
      raise NotImplementedError
    x = action[x_idx]
    y = action[y_idx]
    z = action[z_idx] if z_idx != -1 else self.getPatch_z(24, x, y, rz)

    rot = (rx, ry, rz)

    return motion_primative, x, y, z, rot


  def getPatch_z(self, patch_size, x, y, rz):
        """
        get the image patch in heightmap, centered at center_pixel, rotated by rz
        :param obs: BxCxHxW
        :param center_pixel: Bx2
        :param rz: B
        :return: image patch
        """
        img_size = self.heightmap_size
        row_pixel, column_pixel = self._getPixelsFromPos(x, y)
        center_coordinate = np.array([column_pixel, row_pixel])
        transition = center_coordinate - np.array([self.heightmap_size / 2, self.heightmap_size / 2])
        R = np.asarray([[np.cos(-rz), np.sin(-rz)],
                        [-np.sin(-rz), np.cos(-rz)]])
        rotated_heightmap = rotate(self.heightmap, angle=-rz * 180 / np.pi, reshape=True)
        rotated_transition = R.dot(transition)\
                             + np.array([rotated_heightmap.shape[0] / 2, rotated_heightmap.shape[1] / 2])
        rotated_row_column = np.flip(rotated_transition)
        patch = rotated_heightmap[int(max(rotated_row_column[0] - patch_size / 2, 0)):
                                  int(min(rotated_row_column[0] + patch_size / 2, rotated_heightmap.shape[0])),
                                  int(max(rotated_row_column[1] - 6, 0)):
                                  int(min(rotated_row_column[1] + 6, rotated_heightmap.shape[1]))]
        # print(patch.shape, rotated_row_column)
        z = (np.min(patch) + np.max(patch)) / 2
        gripper_depth = 0.04
        gripper_reach = 0.01
        safe_z_pos = max(z, np.max(patch) - gripper_depth, np.min(patch) + gripper_reach, gripper_reach)
        return safe_z_pos

  def _checkPerfectGrasp(self, x, y, z, rot, objects):
      return True

  def step(self, action):
    pre_obj_grasped = self.obj_grasped
    self.takeAction(action)
    self.wait(100)
    # remove obj that above a threshold hight
    # for obj in self.objects:
    #   if obj.getPosition()[2] > self.pick_pre_offset:
    #     # self.objects.remove(obj)
    #     # pb.removeBody(obj.object_id)
    #     self._removeObject(obj)

    # for obj in self.objects:
    #   if not self._isObjectWithinWorkspace(obj):
    #     self._removeObject(obj)

    obs = self._getObservation(action)
    done = self._checkTermination()
    if self.reward_type == 'dense':
      reward = 1.0 if self.obj_grasped > pre_obj_grasped else 0.0
    else:
      reward = 1.0 if done else 0.0

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    return obs, reward, done

  def reset(self):
    ''''''
    while True:
      self.resetPybulletEnv()
      try:
        if not self.exhibit_env_obj:
            for i in range(self.num_obj):
              x = (np.random.rand() - 0.5) * 0.1
              x += self.workspace[0].mean()
              y = (np.random.rand() - 0.5) * 0.1
              y += self.workspace[1].mean()
              randpos = [x, y, 0.40]
              obj = self._generateShapes(constants.RANDOM_HOUSEHOLD, 1, random_orientation=self.random_orientation,
                                         pos=[randpos], padding=self.min_boarder_padding,
                                         min_distance=self.min_object_distance, model_id=-1)
              # obj = self._generateShapes(constants.RANDOM_HOUSEHOLD200, 1, random_orientation=self.random_orientation,
              #                            pos=[randpos], padding=self.min_boarder_padding,
              #                            min_distance=self.min_object_distance, model_id=-1)
              pb.changeDynamics(obj[0].object_id, -1, lateralFriction=0.6)
              self.wait(100)
        # elif True:
        # #create ducks
        #     for i in range(15):
        #         x = (np.random.rand() - 0.5) * 0.1
        #         x += self.workspace[0].mean()
        #         y = (np.random.rand() - 0.5) * 0.1
        #         y += self.workspace[1].mean()
        #         randpos = [x, y, 0.20]
        #         creat_duck(randpos)
        #         self.wait(100)
        elif self.exhibit_env_obj:  # exhibit all random objects in this environment
            root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
            # urdf_pattern = os.path.join(root_dir, constants.URDF_PATH, 'random_household_object/*/*.urdf')
            urdf_pattern = os.path.join(root_dir, constants.URDF_PATH, 'random_household_object_200/*/*/*.obj')
            found_object_directories = glob.glob(urdf_pattern)
            total_num_objects = len(found_object_directories)

            display_size = 2
            columns = math.ceil(math.sqrt(total_num_objects))
            distance = display_size / (columns - 1)

            for i in range(total_num_objects):
                x = (i // columns) * distance
                x += self.workspace[0].mean() + 0.6
                y = (i % columns) * distance
                y += self.workspace[1].mean() - display_size/2
                display_pos = [x, y, 0.1]
                obj = self._generateShapes(constants.RANDOM_HOUSEHOLD200, 1,
                                           rot=[pb.getQuaternionFromEuler([0., 0., -np.pi/4])],
                                           pos=[display_pos], padding=self.min_boarder_padding,
                                           min_distance=self.min_object_distance, model_id=i)

            self.wait(10000)
      except NoValidPositionException:
        continue
      else:
        break
    self.wait(200)
    self.obj_grasped = 0
    self.num_in_tray_obj = self.num_obj
    return self._getObservation()

  def isObjInBox(self, obj_pos, tray_pos, tray_size):
    tray_range = self.tray_range(tray_pos, tray_size)
    return tray_range[0][0] < obj_pos[0] < tray_range[0][1] and tray_range[1][0] < obj_pos[1] < tray_range[1][1]

  @staticmethod
  def tray_range(tray_pos, tray_size):
    return np.array([[tray_pos[0] - tray_size[0] / 2, tray_pos[0] + tray_size[0] / 2],
                     [tray_pos[1] - tray_size[1] / 2, tray_pos[1] + tray_size[1] / 2]])

  def InBoxObj(self, tray_pos, tray_size):
    obj_list = []
    for obj in self.objects:
      if self.isObjInBox(obj.getPosition(), tray_pos, tray_size):
        obj_list.append(obj)
    return obj_list

  def _checkTermination(self):
    ''''''
    for obj in self.objects:
      # if self._isObjectHeld(obj):
      #   self.obj_grasped += 1
      #   self._removeObject(obj)
      #   if self.obj_grasped == self.num_obj:
      #     return True
      #   return False
      if obj.getPosition()[2] >= 0.35:  #ZXP getPos z > threshold is more robust than _isObjectHeld()
        self.obj_grasped += 1
        self._removeObject(obj)
        if self.obj_grasped == self.num_obj:
          return True
        return False
    return False

  def _getObservation(self, action=None):
    state, in_hand, obs = super(RandomHouseholdPickingClutterEnv, self)._getObservation()
    return 0, np.zeros_like(in_hand), obs

def createRandomHouseholdPickingClutterEnv(config):
  return RandomHouseholdPickingClutterEnv(config)
