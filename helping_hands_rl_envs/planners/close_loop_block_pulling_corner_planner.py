import numpy as np

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.planners.close_loop_planner import CloseLoopPlanner
from helping_hands_rl_envs.pybullet.utils import transformations

class CloseLoopBlockPullingCornerPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.stage = 0 # 1: approaching pre, 2: pre->press 3: pull
    self.current_target = None

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      primitive = constants.PICK_PRIMATIVE if self.current_target[2] is constants.PICK_PRIMATIVE else constants.PLACE_PRIMATIVE
      self.current_target = None
    else:
      primitive = constants.PICK_PRIMATIVE if self.isHolding() else constants.PLACE_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    object_pos = self.env.objects[0].getPosition()
    object_rot = list(transformations.euler_from_quaternion(self.env.objects[0].getRotation()))
    while object_rot[2] > np.pi/2:
      object_rot[2] -= np.pi
    while object_rot[2] < -np.pi/2:
      object_rot[2] += np.pi

    pull_rz = self.env.corner_rz + np.pi/4
    while pull_rz > np.pi/2:
      pull_rz -= np.pi
    while pull_rz < -np.pi/2:
      pull_rz += np.pi

    pre_press_pos = self.env.corner.getPressPose()[0]
    pre_press_pos[2] = 0.15

    pre_press_rot = [0, 0, pull_rz]

    press_pos = self.env.corner.getPressPose()[0]
    press_pos[2] += 0.0375

    pull_pos = self.env.corner.getPullPose()[0]
    pull_pos[2] += 0.0375

    pull_rot = [0, 0, pull_rz]

    post_pull_pos = self.env.corner.getPullPose()[0]
    post_pull_pos[2] = 0.1

    pre_pick_pos = object_pos[0], object_pos[1], 0.1
    if self.stage == 0:
      # moving to pre press
      self.dpos = 0.05
      self.stage = 1
      self.current_target = (pre_press_pos, pre_press_rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 1:
      # moving to press
      self.dpos = 0.01
      self.stage = 2
      self.current_target = (press_pos, pre_press_rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 2:
      # moving to pull
      self.dpos = 0.01
      self.stage = 3
      self.current_target = (pull_pos, pull_rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 3:
      # moving to pre pick
      self.dpos = 0.05
      self.stage = 0
      self.current_target = (post_pull_pos, pull_rot, constants.PLACE_PRIMATIVE)

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.stage = 0
      self.current_target = None
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()

  def getStepsLeft(self):
    return 100