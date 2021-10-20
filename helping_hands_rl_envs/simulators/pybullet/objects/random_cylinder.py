import sys

sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants


class RandomCylinder(PybulletObject):
    def __init__(self, pos, rot, scale):
        radius = np.random.uniform(0.015, 0.025) * scale
        height = np.random.uniform(0.06, 0.12) * scale
        color = np.random.uniform(0.6, 1, (4,))
        color[-1] = 1
        obj_visual = pb.createVisualShape(pb.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        obj_collision = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=radius, height=height)

        object_id = pb.createMultiBody(baseMass=0.15,
                                       baseCollisionShapeIndex=obj_collision,
                                       baseVisualShapeIndex=obj_visual,
                                       basePosition=pos,
                                       baseOrientation=rot)

        pb.changeDynamics(object_id,
                          -1,
                          lateralFriction=1,
                          spinningFriction=0.005,
                          rollingFriction=0.005)

        super(RandomCylinder, self).__init__(constants.RANDOM, object_id)
