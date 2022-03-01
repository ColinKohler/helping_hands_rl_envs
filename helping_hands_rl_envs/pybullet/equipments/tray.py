import os
import pybullet as pb
import numpy as np

import helping_hands_rl_envs
from helping_hands_rl_envs.pybullet.utils import pybullet_util

class Tray:
  def __init__(self):
    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    self.id = None

  def initialize(self, pos=(0, 0, 0), rot=(0, 0, 0, 1), size=(0.2, 0.2, 0.2), color=[0.8, 0.8, 0.8, 1], transparent=False):
    '''
    :param pos:
    :param rot:
    :param size: the maximum box that can be fitted into the outer tray
    :param color:
    :return: tray with walls have inclination, where inclination is the angle between the wall and button normal.
             Box: inclination 0
             Plate: inclination 90
             the inner tray is pink while the outer tray is gray
    '''
    slopes_d = 0.02  # the distance between the inner slops and the outer slops at tray's button
    tray_shrink = 0.03
    size[0] -= tray_shrink
    size[1] -= tray_shrink
    botton_half_thick = 0.05
    half_thickness = 0.001
    pos[2] -= botton_half_thick - 2 * half_thickness
    color_inner = [0.9, 0.6, 0.6, 1]
    if transparent:
      color[3] = 0
      color_inner[3] = 0
    size_inner = size.copy()
    size_outer = size.copy()
    size_inner[0] -= 2 * slopes_d
    size_inner[1] -= 2 * slopes_d
    inclination_inner = np.pi * (70 / 180)
    inclination_outer = np.pi * (45 / 180)
    cos_offset_inner = np.cos(inclination_inner)
    sin_offset_inner = np.sin(inclination_inner)
    cos_offset_outer = np.cos(inclination_outer)
    sin_offset_outer = np.sin(inclination_outer)
    h_inner = slopes_d / (np.tan(inclination_inner) - np.tan(inclination_outer))
    size_inner[2] = h_inner
    half_wall_height_inner = size_inner[2] / (2 * cos_offset_inner)
    half_wall_height_outer = size_outer[2] / (2 * cos_offset_outer)
    size0_inner = size_inner[0] / 2 + 2 * half_wall_height_inner * sin_offset_inner
    size1_inner = size_inner[1] / 2 + 2 * half_wall_height_inner * sin_offset_inner
    size0_outer = size_outer[0] / 2 + 2 * half_wall_height_outer * sin_offset_outer
    size1_outer = size_outer[1] / 2 + 2 * half_wall_height_outer * sin_offset_outer

    bottom_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size0_outer, size1_outer, botton_half_thick],
                                         rgbaColor=color)
    bottom_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size0_outer, size1_outer, botton_half_thick])

    front_visual_inner = pb.createVisualShape(pb.GEOM_BOX,
                                              halfExtents=[half_thickness, size1_inner, half_wall_height_inner],
                                              rgbaColor=color_inner)
    front_collision_inner = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[half_thickness, size1_inner,
                                                                              half_wall_height_inner])

    back_visual_inner = pb.createVisualShape(pb.GEOM_BOX,
                                             halfExtents=[half_thickness, size1_inner, half_wall_height_inner],
                                             rgbaColor=color_inner)
    back_collision_inner = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[half_thickness, size1_inner,
                                                                             half_wall_height_inner])

    left_visual_inner = pb.createVisualShape(pb.GEOM_BOX,
                                             halfExtents=[size0_inner, half_thickness, half_wall_height_inner],
                                             rgbaColor=color_inner)
    left_collision_inner = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size0_inner, half_thickness,
                                                                             half_wall_height_inner])

    right_visual_inner = pb.createVisualShape(pb.GEOM_BOX,
                                              halfExtents=[size0_inner, half_thickness, half_wall_height_inner],
                                              rgbaColor=color_inner)
    right_collision_inner = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size0_inner, half_thickness,
                                                                              half_wall_height_inner])

    front_visual_outer = pb.createVisualShape(pb.GEOM_BOX,
                                              halfExtents=[half_thickness, size1_outer, half_wall_height_outer],
                                              rgbaColor=color)
    front_collision_outer = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[half_thickness, size1_outer,
                                                                              half_wall_height_outer])

    back_visual_outer = pb.createVisualShape(pb.GEOM_BOX,
                                             halfExtents=[half_thickness, size1_outer, half_wall_height_outer],
                                             rgbaColor=color)
    back_collision_outer = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[half_thickness, size1_outer,
                                                                             half_wall_height_outer])

    left_visual_outer = pb.createVisualShape(pb.GEOM_BOX,
                                             halfExtents=[size0_outer, half_thickness, half_wall_height_outer],
                                             rgbaColor=color)
    left_collision_outer = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size0_outer, half_thickness,
                                                                             half_wall_height_outer])

    right_visual_outer = pb.createVisualShape(pb.GEOM_BOX,
                                              halfExtents=[size0_outer, half_thickness, half_wall_height_outer],
                                              rgbaColor=color)
    right_collision_outer = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size0_outer, half_thickness,
                                                                              half_wall_height_outer])

    self.id = pb.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=bottom_collision,
                                 baseVisualShapeIndex=bottom_visual,
                                 basePosition=pos,
                                 baseOrientation=rot,
                                 linkMasses=[1, 1, 1, 1, 1, 1, 1, 1],
                                 linkCollisionShapeIndices=[front_collision_inner, back_collision_inner,
                                                            left_collision_inner, right_collision_inner,
                                                            front_collision_outer, back_collision_outer,
                                                            left_collision_outer, right_collision_outer],
                                 linkVisualShapeIndices=[front_visual_inner, back_visual_inner,
                                                         left_visual_inner, right_visual_inner,
                                                         front_visual_outer, back_visual_outer,
                                                         left_visual_outer, right_visual_outer],
                                 linkPositions=[[-size_inner[0] / 2 - sin_offset_inner * half_wall_height_inner, 0,
                                                 size_inner[2] / 2 + botton_half_thick],
                                                [size_inner[0] / 2 + sin_offset_inner * half_wall_height_inner, 0,
                                                 size_inner[2] / 2 + botton_half_thick],
                                                [0, -size_inner[1] / 2 - sin_offset_inner * half_wall_height_inner,
                                                 size_inner[2] / 2 + botton_half_thick],
                                                [0, size_inner[1] / 2 + sin_offset_inner * half_wall_height_inner,
                                                 size_inner[2] / 2 + botton_half_thick],
                                                [-size_outer[0] / 2 - sin_offset_outer * half_wall_height_outer, 0,
                                                 size_outer[2] / 2 + botton_half_thick],
                                                [size_outer[0] / 2 + sin_offset_outer * half_wall_height_outer, 0,
                                                 size_outer[2] / 2 + botton_half_thick],
                                                [0, -size_outer[1] / 2 - sin_offset_outer * half_wall_height_outer,
                                                 size_outer[2] / 2 + botton_half_thick],
                                                [0, size_outer[1] / 2 + sin_offset_outer * half_wall_height_outer,
                                                 size_outer[2] / 2 + botton_half_thick]],
                                 linkOrientations=[pb.getQuaternionFromEuler([0., -inclination_inner, 0.]),
                                                   pb.getQuaternionFromEuler([0., inclination_inner, 0.]),
                                                   pb.getQuaternionFromEuler([inclination_inner, 0., 0.]),
                                                   pb.getQuaternionFromEuler([-inclination_inner, 0., 0.]),
                                                   pb.getQuaternionFromEuler([0., -inclination_outer, 0.]),
                                                   pb.getQuaternionFromEuler([0., inclination_outer, 0.]),
                                                   pb.getQuaternionFromEuler([inclination_outer, 0., 0.]),
                                                   pb.getQuaternionFromEuler([-inclination_outer, 0., 0.])],
                                 linkInertialFramePositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                                                             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                 linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
                                                                [0, 0, 0, 1],
                                                                [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
                                                                [0, 0, 0, 1]],
                                 linkParentIndices=[0, 0, 0, 0, 0, 0, 0, 0],
                                 linkJointTypes=[pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED,
                                                 pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED],
                                 linkJointAxis=[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
                                                [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
                                 )
    for i in range(1, 9):
      pb.changeDynamics(self.id,
                        i,
                        lateralFriction=0.,
                        rollingFriction=0.,
                        linearDamping=0.1)

    pb.changeDynamics(self.id,
                      -1,
                      lateralFriction=0.8,
                      rollingFriction=0.01,
                      linearDamping=0.1)

  def reset(self, pos=(0, 0, 0), rot=(0, 0, 0, 1)):
    pb.resetBasePositionAndOrientation(self.id, pos, rot)

  def remove(self):
    if self.id:
      pb.removeBody(self.id)
    self.id = None
