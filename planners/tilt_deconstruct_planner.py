import numpy as np
import numpy.random as npr
import pybullet as pb
from itertools import combinations

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class TiltDeconstructPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(TiltDeconstructPlanner, self).__init__(env, config)
    self.objs_to_remove = []

  def getStepLeft(self):
    return 100

  def pickTallestObjOnTop(self, objects=None, side_grasp=False):
    """
    pick up the highest object that is on top
    :param objects: pool of objects
    :param side_grasp: grasp on the side of the object (90 degree), should be true for triangle, brick, etc
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(objects=objects)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2]+self.env.pick_offset, object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2]+self.env.pick_offset, pose[5]
        if obj in self.objs_to_remove:
          self.objs_to_remove.remove(obj)
        if self.env.object_types[obj] in [constants.ROOF, constants.TRIANGLE, constants.BRICK]:
          side_grasp = True
        break
    if side_grasp:
      r += np.pi / 2
      while r < 0:
        r += 2*np.pi
      while r > 2*np.pi:
        r -= 2*np.pi
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def placeOnGround(self, padding_dist, min_dist):
    """
    place on the ground, avoiding all existing objects
    :param padding_dist: padding dist for getting valid pos
    :param min_dist: min dist to adjacent object
    :return: encoded action
    """
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x), self.env.objects))]
    # if np.random.random() > 0.5:
    #   sample_range = [self.env.workspace[0], [self.env.tilt_border + 0.02, self.env.workspace[1][1]]]
    # else:
    #   sample_range = [self.env.workspace[0], [self.env.workspace[1][0], self.env.tilt_border2 - 0.02]]
    # try:
    #   place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1, sample_range)[0]
    # except NoValidPositionException:
    #   try:
    #     place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1)[0]
    #   except NoValidPositionException:
    #     place_pos = self.getValidPositions(padding_dist, min_dist, [], 1)[0]
    try:
      place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1)[0]
    except NoValidPositionException:
      place_pos = self.getValidPositions(padding_dist, min_dist, [], 1)[0]
    x, y, z, rz = place_pos[0], place_pos[1], self.env.place_offset, 0
    if place_pos[1] < self.env.tilt_border2:
      rx = -self.env.tilt_plain2_rx
    elif place_pos[1] > self.env.tilt_border:
      rx = -self.env.tilt_plain_rx
    else:
      rx = 0
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, rx))

  def getPickingAction(self):
    if self.env.checkStructure():
      self.objs_to_remove = [o for o in self.env.structure_objs]
    if not self.objs_to_remove:
      return self.pickTallestObjOnTop()
    return self.pickTallestObjOnTop(self.objs_to_remove)

  def getPlacingAction(self):
    return self.placeOnGround(self.env.max_block_size * 2, self.env.max_block_size * 2.7)
