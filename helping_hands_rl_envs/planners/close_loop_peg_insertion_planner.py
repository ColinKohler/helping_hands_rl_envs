import numpy as np

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.planners.close_loop_planner import CloseLoopPlanner
from helping_hands_rl_envs.pybullet.utils import transformations

class CloseLoopPegInsertionPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.stage = 0
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
    pre_insert_pos, pre_insert_rot = self.env.peg_hole.getHolePose()
    pre_insert_pos[2] += 0.12
    pre_insert_rot = transformations.euler_from_quaternion(pre_insert_rot)

    insert_pos, insert_rot = self.env.peg_hole.getHolePose()
    insert_rot = transformations.euler_from_quaternion(insert_rot)

    if self.stage == 0:
      # moving to pre insert
      self.stage = 1
      self.current_target = (pre_insert_pos, pre_insert_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 1:
      # insert peg
      self.stage = 0
      self.current_target = (insert_pos, insert_rot, constants.PICK_PRIMATIVE)

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
