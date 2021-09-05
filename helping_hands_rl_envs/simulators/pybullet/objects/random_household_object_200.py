import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os
import glob

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.random_household_object_200_info import obj_scales, obj_centers, obj_avg_sr

obj_pattern = os.path.join(os.path.dirname(helping_hands_rl_envs.__file__),
                           'simulators/urdf/object/random_household_object_200/3dnet/*/*.obj')
found_object_directories = sorted(glob.glob(obj_pattern))
total_num_objects = len(found_object_directories)

class RandomHouseHoldObject200:
    def __init__(self, pos, rot, scale=0.5, index=-1):
        if index < 0:
            while True:
                index = np.random.choice(np.arange(total_num_objects), 1)[0]
                if obj_avg_sr[index] > 0.97:  # sr > 0.8:  143 objs; sr > 0.9: 134 objs;
                                              # sr > 0.95: 105 objs; sr > 0.99: 41 objs;
                                              # sr > 0.97: 76 objs
                                              # avg sr = 0.923; avg sr > 0.8 = 0.968
                                              # avg sr > 0.97 = 0.9938
                    break
        else:
            # sorted_obj_sr_indx = np.argsort(obj_avg_sr)[::-1]
            pass
            # index = sorted_obj_sr_indx[index]
        obj_filepath = found_object_directories[index]

        color = np.random.uniform(0.6, 1, (4,))
        color[-1] = 1

        real_scale = scale * obj_scales[index]
        center = obj_centers[index]
        obj_visual = pb.createVisualShape(pb.GEOM_MESH,
                                          fileName=obj_filepath,
                                          meshScale=[real_scale, real_scale, real_scale],
                                          rgbaColor=color,
                                          visualFramePosition=center)
        obj_collision = pb.createCollisionShape(pb.GEOM_MESH,
                                                fileName=obj_filepath,
                                                meshScale=[real_scale, real_scale, real_scale],
                                                collisionFramePosition=center)
        self.center = center
        self.real_scale = real_scale

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

        self.object_id = object_id

    def getAABB(self):
        return np.array(pb.getAABB(self.id_, -1))

    def let_fall(self, valid_position_checker, max_iters=50):
        for _ in range(max_iters):
            [pb.stepSimulation() for _ in range(10)]
            # COM = pb.getBasePositionAndOrientation(self.id_)[0][:2]
            COM = np.mean(self.getAABB()[:,:2], axis=0)
            if not valid_position_checker(COM):
                return False
            vel = np.concatenate(pb.getBaseVelocity(self.id_)[-2:])
            if (np.abs(vel) < 0.001).all():
                return True
        return False

    def get_position(self):
        return pb.getBasePositionAndOrientation(self.id_)[0]

    @property
    def id_(self):
        return self.object_id

if __name__ == "__main__":
    fit_to_gripper(0.05)
