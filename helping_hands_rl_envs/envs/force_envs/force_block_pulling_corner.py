import numpy as np
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_pulling_corner import CloseLoopBlockPullingCornerEnv

class ForceBlockPullingCornerEnv(CloseLoopBlockPullingCornerEnv):
  def __init__(self, config):
    super().__init__(config)
    self.prev_force = None

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)

    #finger_a_force, finger_b_force = self.robot.getFingerForce()
    #finger_force = [np.sqrt(np.sum(finger_a_force[:3])**2), np.sqrt(np.sum(finger_b_force[:3])**2)]
    #finger_force = np.array([finger_a_force[:3], finger_b_force[:3]]).reshape(-1)

    force, moment = self.robot.getWristForce()
    force = np.concatenate((force, moment))
    #diff_force = force - self.prev_force if self.prev_force is not None else force

    self.prev_force = force
    return state, hand_obs, obs, force

def createForceBlockPullingCornerEnv(config):
  return ForceBlockPullingCornerEnv(config)
