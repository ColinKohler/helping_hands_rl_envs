import sys

sys.path.append('..')

import pybullet as pb
import numpy as np
import os
import glob

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants

root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
# obj_pattern = os.path.join(root_dir, constants.URDF_PATH, 'GraspNet1B_object_textured/0*/')
obj_pattern = os.path.join(root_dir, constants.URDF_PATH, 'GraspNet1B_object/0*/')
found_object_directories = sorted(glob.glob(obj_pattern))
total_num_objects = len(found_object_directories)

COLORS = [[0.83359415, 0.99723381, 0.99342488, 0.90968221],
          [0.85932704, 0.65041388, 0.68460488, 0.91188582],
          [0.66331642, 0.92962455, 0.61745929, 0.66165534],
          [0.85049655, 0.93082582, 0.80304686, 0.92337727],
          [0.60827337, 0.87545109, 0.84094463, 0.69882815],
          [0.68512538, 0.87459948, 0.6573139, 0.78492144],
          [0.79982731, 0.87262376, 0.98871526, 0.97483433],
          [0.96483257, 0.95125105, 0.64371884, 0.79836311],
          [0.889349, 0.63974198, 0.94567148, 0.87870439],
          [0.70580075, 0.72565715, 0.76486561, 0.7731156],
          [0.93095998, 0.81018395, 0.75564586, 0.67677224],
          [0.75494942, 0.77817516, 0.70138636, 0.88011722],
          [0.63972335, 0.8234283, 0.75937976, 0.89927497],
          [0.97073293, 0.77845947, 0.86038355, 0.82634558],
          [0.60903538, 0.97452814, 0.62657003, 0.82749687],
          [0.84524216, 0.77766505, 0.68959266, 0.88670976],
          [0.82248481, 0.61341774, 0.78994926, 0.71266927],
          [0.86431542, 0.88068822, 0.81929904, 0.70107859],
          [0.91712835, 0.64712152, 0.80678721, 0.93559243],
          [0.93035074, 0.72947959, 0.73771991, 0.60197532],
          [0.69375461, 0.60385541, 0.8150952, 0.93569609],
          [0.61376728, 0.70884089, 0.92944799, 0.6205812],
          [0.66874295, 0.67732346, 0.69847895, 0.60972561],
          [0.820284, 0.66196911, 0.65504595, 0.89031297],
          [0.73414285, 0.77491158, 0.62997718, 0.99092358],
          [0.8120352, 0.64512029, 0.6189169, 0.72403261],
          [0.8218499, 0.72457316, 0.81985337, 0.78104482],
          [0.68337914, 0.6102489, 0.76937169, 0.67252713],
          [0.65442649, 0.66138339, 0.61596737, 0.63814213],
          [0.85059767, 0.72553581, 0.6496602, 0.83470813],
          [0.91256462, 0.97108811, 0.7402825, 0.99529857],
          [0.63291895, 0.97728718, 0.86927159, 0.65785246],
          [0.97837238, 0.95931874, 0.91799905, 0.83788678],
          [0.78694082, 0.66566154, 0.67904726, 0.60290477],
          [0.62222487, 0.92952348, 0.8130582, 0.81580428],
          [0.95294782, 0.70855411, 0.70436731, 0.89833003],
          [0.68085334, 0.7360199, 0.83977906, 0.87220048],
          [0.62979706, 0.68430129, 0.94193875, 0.83987562],
          [0.92699273, 0.6332265, 0.71286961, 0.98919942],
          [0.69485776, 0.79395019, 0.73103725, 0.81593225],
          [0.76758362, 0.60118147, 0.90805406, 0.98320474],
          [0.72064699, 0.83543488, 0.63921461, 0.70029478],
          [0.91765848, 0.90539355, 0.89293851, 0.87120133],
          [0.94560148, 0.91140139, 0.72112741, 0.66431116],
          [0.91665862, 0.62079201, 0.92569548, 0.77149855],
          [0.67717598, 0.63473647, 0.95576286, 0.97975469],
          [0.71695299, 0.86795012, 0.82534581, 0.64409321],
          [0.83941257, 0.67327794, 0.91950692, 0.74232045],
          [0.72899728, 0.90030947, 0.69607097, 0.81482407],
          [0.71026322, 0.7601246, 0.63228006, 0.75182753],
          [0.7969585, 0.89405846, 0.66699145, 0.61604387],
          [0.98602134, 0.69607393, 0.88646021, 0.80765027],
          [0.73317678, 0.8974576, 0.62069225, 0.75132512],
          [0.94456367, 0.90094407, 0.82519716, 0.85315353],
          [0.77569957, 0.68957297, 0.74927008, 0.76779493],
          [0.75125458, 0.79445436, 0.83438545, 0.68654407],
          [0.74642898, 0.86148109, 0.6618111, 0.67878009],
          [0.68821182, 0.70498948, 0.75638712, 0.77569209],
          [0.97472475, 0.80311909, 0.98678622, 0.75874153],
          [0.83999728, 0.9905068, 0.6415373, 0.92889649],
          [0.86118119, 0.64207568, 0.94504008, 0.90739138],
          [0.60925236, 0.64610776, 0.92648858, 0.94666514],
          [0.76472032, 0.60199382, 0.95858124, 0.88282233],
          [0.98128663, 0.60963179, 0.69026466, 0.60985517],
          [0.73490587, 0.79640618, 0.86511689, 0.87995896],
          [0.65547622, 0.66738026, 0.65795893, 0.70217417],
          [0.6198114, 0.97025389, 0.91479966, 0.95876779],
          [0.85730812, 0.77707335, 0.70277407, 0.61668665],
          [0.7151256, 0.98489724, 0.85934228, 0.80644446],
          [0.94750232, 0.93661664, 0.66717491, 0.7192019],
          [0.83534209, 0.997451, 0.64551032, 0.82590201],
          [0.90157418, 0.67783242, 0.9787091, 0.72618622],
          [0.79713991, 0.9829045, 0.78435307, 0.99846299],
          [0.87065756, 0.81155579, 0.69176534, 0.63701609],
          [0.95639873, 0.64264744, 0.75521895, 0.94655509],
          [0.82433228, 0.60661764, 0.75898621, 0.89301617],
          [0.97905441, 0.78566563, 0.68761669, 0.64195916],
          [0.99994282, 0.73087481, 0.74982151, 0.63545526],
          [0.99236466, 0.82175127, 0.97736902, 0.83350699],
          [0.68880914, 0.78566831, 0.70368, 0.79031851],
          [0.66138191, 0.67232898, 0.63368974, 0.99745585],
          [0.99843684, 0.69629567, 0.78500517, 0.82926765],
          [0.75426557, 0.76028977, 0.97260676, 0.89466544],
          [0.95503496, 0.8740204, 0.70159072, 0.69475871],
          [0.926544, 0.68217962, 0.82616935, 0.98284703],
          [0.7562463, 0.89397063, 0.83576639, 0.92347182],
          [0.98337972, 0.87453535, 0.96873198, 0.92452792],
          [0.8685709, 0.84361064, 0.77971926, 0.71857389],
          [0.87145502, 0.82611949, 0.96593294, 0.71522765],
          [0.81767836, 0.91607953, 0.67799341, 0.82077175],
          [0.87331301, 0.79233829, 0.8723416, 0.74931595],
          [0.72955552, 0.69690454, 0.9462561, 0.93086011],
          [0.68824667, 0.60452654, 0.81916299, 0.65425155],
          [0.84334435, 0.83942558, 0.82388211, 0.7818232],
          [0.68518089, 0.87470875, 0.67245383, 0.97237673],
          [0.65805713, 0.89899799, 0.82564458, 0.66516921],
          [0.89665463, 0.62213256, 0.99757297, 0.64553824],
          [0.99598952, 0.72111567, 0.8764059, 0.72369218],
          [0.78513214, 0.893123, 0.92771135, 0.88705852],
          [0.90398539, 0.80781921, 0.74418058, 0.77652107]]


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


class GraspNetObject(PybulletObject):
    def __init__(self, pos, rot, scale, index=-1, alpha=None, colors=None):

        if index >= 0:
            obj_filepath = found_object_directories[index]
        else:
            index = np.random.choice(np.arange(total_num_objects), 1)[0]
            obj_filepath = found_object_directories[index]

        if colors is None or colors == 'random':
            color = np.random.uniform(0.6, 1, (4,))
        elif colors == 'constant':
            color = COLORS[0]
        elif colors == 'assigned':
            color = COLORS[index]
        elif colors == 'white':
            color = [1, 1, 1, 1]
        else:
            raise NotImplementedError(colors)
        color[-1] = 1 if alpha is None else alpha

        self.center = [0, 0, 0]
        obj_edge_max = 0.15 * scale  # the maximum edge size of an obj before scaling
        obj_edge_min = 0.014 * scale  # the minimum edge size of an obj before scaling
        obj_volume_max = 0.0006 * (scale ** 3)  # the maximum volume of an obj before scaling
        obj_scale = scale

        while True:
            obj_visual = pb.createVisualShape(pb.GEOM_MESH,
                                              fileName=obj_filepath + 'convex.obj',
                                              rgbaColor=color,
                                              meshScale=[obj_scale, obj_scale, obj_scale])
            obj_collision = pb.createCollisionShape(pb.GEOM_MESH,
                                                    fileName=obj_filepath + 'convex.obj',
                                                    meshScale=[obj_scale, obj_scale, obj_scale])

            object_id = pb.createMultiBody(baseMass=0.15,
                                           baseCollisionShapeIndex=obj_collision,
                                           baseVisualShapeIndex=obj_visual,
                                           basePosition=pos,
                                           baseOrientation=rot)

            aabb = pb.getAABB(object_id)
            aabb = np.asarray(aabb)
            size = aabb[1] - aabb[0]

            if np.partition(size, -2)[-2] > obj_edge_max:
                obj_scale *= 0.8
                pb.removeBody(object_id)
            elif size[0] * size[1] * size[2] > obj_volume_max:
                obj_scale *= 0.85
                pb.removeBody(object_id)
            elif size.min() < obj_edge_min:
                obj_scale /= 0.95
                pb.removeBody(object_id)
            else:
                pb.removeBody(object_id)
                break

        # obj_visual = pb.createVisualShape(pb.GEOM_MESH,
        #                                   fileName=obj_filepath + 'textured.obj',
        #                                   rgbaColor=color,
        #                                   meshScale=[obj_scale, obj_scale, obj_scale])
        obj_visual = pb.createVisualShape(pb.GEOM_MESH,
                                          fileName=obj_filepath + 'convex.obj',
                                          rgbaColor=color,
                                          meshScale=[obj_scale, obj_scale, obj_scale])
        obj_collision = pb.createCollisionShape(pb.GEOM_MESH,
                                                fileName=obj_filepath + 'convex.obj',
                                                meshScale=[obj_scale, obj_scale, obj_scale])

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
