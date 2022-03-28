import os
import math
import glob
import numpy as np
import torch
from matplotlib import pyplot as plt

import helping_hands_rl_envs
from helping_hands_rl_envs.envs.base_env import BaseEnv
from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.pybullet.utils.constants import NoValidPositionException
from helping_hands_rl_envs.pybullet.equipments.tray import Tray
from scipy.ndimage.interpolation import rotate
import pybullet as pb


def creat_duck(pos):
    shift = [0, -0.02, 0]
    scale = 0.08
    meshScale = [scale, scale, scale]
    # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
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


class FiveDGrasping(BaseEnv):
    '''
  '''

    def __init__(self, config):
        super(FiveDGrasping, self).__init__(config)
        self.object_init_z = 0.1
        self.obj_grasped = 0
        self.tray = Tray()
        self.exhibit_env_obj = False
        # self.exhibit_env_obj = True
        self.z_heuristic = config['z_heuristic']
        self.bin_size = config['bin_size']
        self.uncalibrated = config['uncalibrated']
        self.gripper_depth = 0.04
        self.gripper_reach = 0.01

    def initialize(self):
        super().initialize()

    def _decodeAction(self, action, return_collision_z=False):
        """
        decode input action base on self.action_sequence
        Args:
          action: action tensor

        Returns: motion_primative, x, y, z, rot

        """
        primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a),
                                                          ['p', 'x', 'y', 'z', 'r'])
        motion_primative = action[primative_idx] if primative_idx != -1 else 0
        if self.action_sequence.count('r') <= 1:
            rz = action[rot_idx] if rot_idx != -1 else 0
            ry = 0
            rx = 0
        else:
            raise NotImplementedError
        x = action[x_idx]
        y = action[y_idx]

        if z_idx != -1:
            z = action[z_idx]
        else:
            z = self.getPatch_z(x, y, rz)

        rot = (rx, ry, rz)

        if return_collision_z:
            tray_bottom = self.getPatch_z(x, y, rz, return_collision_z=True)
            return motion_primative, x, y, z, rot, tray_bottom

        return motion_primative, x, y, z, rot

    def _getPixelsFromPos(self, x, y):
        row_pixel, col_pixel = super()._getPixelsFromPos(x, y)
        row_pixel = min(row_pixel, self.heightmap_size - self.in_hand_size / 2 - 1)
        row_pixel = max(row_pixel, self.in_hand_size / 2)
        col_pixel = min(col_pixel, self.heightmap_size - self.in_hand_size / 2 - 1)
        col_pixel = max(col_pixel, self.in_hand_size / 2)
        return row_pixel, col_pixel

    def getPatch_z(self, x, y, rz, z=None, return_collision_z=False):
        """
        get the image patch in heightmap, centered at center_pixel, rotated by rz
        :param obs:
        :param center_pixel
        :param rz:
        :return: safe z
        """
        row_pixel, col_pixel = self._getPixelsFromPos(x, y)
        # local_region is as large as ih_img
        local_region = self.heightmap[int(row_pixel - self.in_hand_size / 2): int(row_pixel + self.in_hand_size / 2),
                       int(col_pixel - self.in_hand_size / 2): int(col_pixel + self.in_hand_size / 2)]
        local_region = rotate(local_region, angle=-rz * 180 / np.pi, reshape=False, mode='nearest')
        patch = local_region[int(self.in_hand_size / 2 - 16):int(self.in_hand_size / 2 + 16),
                int(self.in_hand_size / 2 - 4):int(self.in_hand_size / 2 + 4)]

        # tray_bottom_z_pos = np.mean(patch.flatten()[(patch).flatten().argsort()[2:12]]) + self.gripper_reach

        # tray_bottom_z_pos is the minimum height of the grasp_pose_edge
        # collision_z_pos is the maximum height of the grasp_pose_edge
        grasp_pose_edge = np.zeros((8, 8))
        grasp_pose_edge[0:4, :] = patch[0:4, :]
        grasp_pose_edge[-4:, :] = patch[-4:, :]
        # collision_z_pos = np.mean(grasp_pose_edge.flatten()[(-grasp_pose_edge).flatten().argsort()[2:8]]) \
        #                     + self.gripper_reach
        tray_bottom_z_pos = np.mean(grasp_pose_edge.flatten()[(grasp_pose_edge).flatten().argsort()[2:8]]) \
                            + self.gripper_reach
        if z is None:
            aggressive_z_pos = np.mean(patch.flatten()[(-patch).flatten().argsort()[2:12]]) - self.gripper_depth
            safe_z_pos = max(aggressive_z_pos, tray_bottom_z_pos)
        else:
            safe_z_pos = np.mean(patch.flatten()[(-patch).flatten().argsort()[2:12]]) + z

        # use clearance to prevent gripper colliding with ground
        # safe_z_pos = max(safe_z_pos, self.workspace[2, 0] + self.gripper_reach)
        # safe_z_pos = min(safe_z_pos, self.workspace[2, 1])
        # assert self.workspace[2][0] <= safe_z_pos <= self.workspace[2][1]

        if return_collision_z:
            return tray_bottom_z_pos
        return safe_z_pos

    def _checkPerfectGrasp(self, x, y, z, rot, objects):
        return True

    def step(self, action):
        pre_obj_grasped = self.obj_grasped
        if self.reward_type == 'dense_scene':
            obs0 = self._getObservation()
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

        obs1 = self._getObservation()
        done = self._checkTermination()

        if self.reward_type == 'dense_scene':
            # the difference d measures the change of the observation outside the grasp local in terns of pixel
            # value. Where the grasp local is the region within radius of the gripper.
            # d in [0, 1]
            motion_primative, x, y, z, rot, grasp_local_bottom = self._decodeAction(action, return_collision_z=True)
            row_pixel, col_pixel = self._getPixelsFromPos(x, y)

            xs = np.arange(-(self.in_hand_size - 1) / 2, (self.in_hand_size - 1) / 2 + 1, 1).reshape(1, -1).repeat(
                self.in_hand_size, axis=0)
            ys = np.arange(-(self.in_hand_size - 1) / 2, (self.in_hand_size - 1) / 2 + 1, 1).reshape(-1, 1).repeat(
                self.in_hand_size, axis=1)
            circle = (np.power(xs, 2) + np.power(ys, 2)) <= (self.in_hand_size / 2) ** 2

            s0 = obs0[2][0]
            s1 = obs1[2][0]

            grasp_local = np.zeros_like(s0, dtype=bool)
            grasp_local[int(row_pixel - self.in_hand_size / 2):int(row_pixel + self.in_hand_size / 2),
                        int(col_pixel - self.in_hand_size / 2):int(col_pixel + self.in_hand_size / 2)] = circle
            outside = np.logical_not(grasp_local)

            # d_scene: the difference of the scene
            # d_collision: the distance z goes below grasp_local_bottom
            d_scene = (np.abs(s0[outside] - s1[outside]) / np.maximum(s0[outside], s1[outside])).mean()
            d_collision = np.maximum(grasp_local_bottom - z, 0)
            d = d_scene * 10 + d_collision * 50
            # d = d_collision * 50
            # d = min(d, 0.499)
            d = min(d, 1)

            # plt.figure()
            # plt.imshow(mask.astype(float))
            # plt.colorbar()
            #
            # dx = np.abs(s0 - s1) / np.maximum(s0, s1)
            # plt.figure()
            # plt.imshow(dx)
            # plt.colorbar()
            #
            # plt.figure()
            # plt.imshow(s0)
            # plt.colorbar()
            #
            # plt.figure()
            # plt.imshow(s1)
            # plt.colorbar()
            # plt.show()

        self.robot.closeGripper()
        if self.reward_type == 'dense':
            reward = 1.0 if self.obj_grasped > pre_obj_grasped else 0
        elif self.reward_type == 'dense_scene':
            reward = 1.0 - d if self.obj_grasped > pre_obj_grasped else 0
        else:
            reward = 1.0 if done else 0.0

        if not done:
            done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
        self.current_episode_steps += 1

        return obs1, reward, done

    def isSimValid(self):
        if self.robot.getGripperOpenRatio() > 1:
            return False
        for obj in self.objects:
            p = obj.getPosition()
            if not self.check_random_obj_valid and self.object_types[obj] == constants.RANDOM:
                continue
            if obj.getPosition()[2] >= 0.35 or self._isObjectHeld(obj):
                continue
            if self.workspace_check == 'point':
                if not self._isPointInWorkspace(p):
                    return False
            else:
                if not self._isObjectWithinWorkspace(obj):
                    return False
            if self.pos_candidate is not None:
                if np.abs(self.pos_candidate[0] - p[0]).min() > 0.02 or np.abs(
                        self.pos_candidate[1] - p[1]).min() > 0.02:
                    return False
        return True

    def reset(self):
        ''''''
        while True:
            self.resetPybulletWorkspace()
            if self.tray.id is not None:
                self.tray.remove()
            if self.uncalibrated == 'z':
                tray_z = np.random.uniform(0.1, 0.2)
                tray_rot = np.random.uniform(0, 0, 3)
            elif self.uncalibrated == 'r':
                tray_z = np.random.uniform(0., 0.)
                tray_rot = np.random.uniform(-0.157, 0.157, 3)
            elif self.uncalibrated == 'z_r':
                tray_z = np.random.uniform(0.1, 0.2)
                tray_rot = np.random.uniform(-0.157, 0.157, 3)
            else:
                tray_z = np.random.uniform(0., 0.)
                tray_rot = np.random.uniform(0, 0, 3)
            # tray_z = np.random.uniform(0.05, 0.2)
            # tray_rot = np.random.uniform(-0.314, 0.314, 3)
            tray_rot[-1] = np.asarray([0])
            self.tray_pos = pb.getQuaternionFromEuler(tray_rot)
            self.tray.initialize(pos=[self.workspace[0].mean(), self.workspace[1].mean(), tray_z],
                                 rot=self.tray_pos, transparent=True,
                                 size=[self.bin_size, self.bin_size, 0.2])

            try:
                if not self.exhibit_env_obj:
                    for i in range(self.num_obj):
                        x = (np.random.rand() - 0.5) * 0.1
                        x += self.workspace[0].mean()
                        y = (np.random.rand() - 0.5) * 0.1
                        y += self.workspace[1].mean()
                        randpos = [x, y, 0.40 + tray_z]
                        # obj = self._generateShapes(constants.RANDOM_HOUSEHOLD, 1, random_orientation=self.random_orientation,
                        #                            pos=[randpos], padding=self.min_boarder_padding,
                        #                            min_distance=self.min_object_distance, model_id=-1)
                        # obj = self._generateShapes(constants.RANDOM_HOUSEHOLD200, 1,
                        #                            random_orientation=self.random_orientation,
                        #                            pos=[randpos], padding=self.min_boarder_padding,
                        #                            min_distance=self.min_object_distance, model_id=-1)

                        obj = self._generateShapes(constants.GRASP_NET_OBJ, 1,
                                                   random_orientation=self.random_orientation,
                                                   pos=[randpos], padding=self.min_boarder_padding,
                                                   min_distance=self.min_object_distance, model_id=-1)
                        pb.changeDynamics(obj[0].object_id, -1, lateralFriction=0.6)
                        self.wait(10)
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
                    # root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
                    # urdf_pattern = os.path.join(root_dir, constants.URDF_PATH, 'random_household_object_200/*/*/*.obj')
                    # found_object_directories = glob.glob(urdf_pattern)
                    # total_num_objects = len(found_object_directories)
                    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
                    urdf_pattern = os.path.join(root_dir, constants.URDF_PATH, 'object/GraspNet1B_object/0*/')
                    found_object_directories = glob.glob(urdf_pattern)
                    total_num_objects = len(found_object_directories)

                    display_size = 1.5
                    columns = math.floor(math.sqrt(total_num_objects))
                    distance = display_size / (columns - 1)

                    obj_centers = []
                    obj_scales = []

                    for i in range(total_num_objects):
                        x = (i // columns) * distance
                        x += self.workspace[0].mean() + 0.6
                        y = (i % columns) * distance
                        y += self.workspace[1].mean() - display_size / 2
                        display_pos = [x, y, 0.08]
                        obj = self._generateShapes(constants.GRASP_NET_OBJ, 1,
                                                   rot=[pb.getQuaternionFromEuler([0., 0., -np.pi / 4])],
                                                   pos=[display_pos], padding=self.min_boarder_padding,
                                                   min_distance=self.min_object_distance, model_id=i)
                        # obj = self._generateShapes(constants.RANDOM_HOUSEHOLD200, 1,
                        #                            rot=[pb.getQuaternionFromEuler([0., 0., -np.pi / 4])],
                        #                            pos=[display_pos], padding=self.min_boarder_padding,
                        #                            min_distance=self.min_object_distance, model_id=i)
                    #     obj_centers.append(obj[0].center)
                    #     obj_scales.append(obj[0].real_scale)
                    #
                    # obj_centers = np.array(obj_centers)
                    # obj_scales = np.array(obj_scales)
                    print('Number of all objects: ', total_num_objects)
                    self.wait(10000)
            except NoValidPositionException:
                continue
            else:
                break
        self.wait(200)
        self.obj_grasped = 0
        # self.num_in_tray_obj = self.num_obj
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
            if obj.getPosition()[2] >= 0.35 or self._isObjectHeld(obj):
                # ZXP getPos z > threshold is more robust than _isObjectHeld()
                self.obj_grasped += 1
                self._removeObject(obj)
                if self.obj_grasped == self.num_obj or len(self.objects) == 0:
                    return True
                return False
        return False

    def _getObservation(self, action=None):
        state, in_hand, obs = super(FiveDGrasping, self)._getObservation()
        return 0, np.zeros_like(in_hand), obs


def createFiveDGrasping(config):
    return FiveDGrasping(config)
