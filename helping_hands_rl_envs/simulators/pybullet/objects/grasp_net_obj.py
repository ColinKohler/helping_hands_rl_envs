import sys

sys.path.append('..')

import pybullet as pb
import numpy as np
import os
import glob
import re

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.objects.random_household_object_200_info import *

root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
obj_pattern = os.path.join(root_dir, constants.URDF_PATH, 'GraspNet1B_object/0*/')
found_object_directories = sorted(glob.glob(obj_pattern))
total_num_objects = len(found_object_directories)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


class GraspNetObject(PybulletObject):
    def __init__(self, pos, rot, scale, index=-1):

        if index:
            obj_filepath = found_object_directories[index]
        else:
            index = np.random.choice(np.arange(total_num_objects), 1)[0]
            obj_filepath = found_object_directories[index]


        color = np.random.uniform(0.6, 1, (4,))
        color[-1] = 1

        obj_visual = pb.createVisualShape(pb.GEOM_MESH,
                                          fileName=obj_filepath + 'convex.obj',
                                          rgbaColor=color,
                                          meshScale=[scale, scale, scale])
        obj_collision = pb.createCollisionShape(pb.GEOM_MESH,
                                                fileName=obj_filepath + 'convex.obj',
                                                meshScale=[scale, scale, scale])
        self.center = [0, 0, 0]

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

        super(GraspNetObject, self).__init__(constants.RANDOM, object_id)
