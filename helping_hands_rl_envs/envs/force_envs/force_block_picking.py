import numpy as np
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_picking import CloseLoopBlockPickingEnv

class ForceBlockPickingEnv(CloseLoopBlockPickingEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)
    finger_a_force, finger_b_force = self.robot.getFingerForceVector()

    finger_force = [np.sqrt(np.sum(finger_a_force)**2), np.sqrt(np.sum(finger_a_force)**2)]

    return state, hand_obs, obs, np.array(finger_force)

def createForceBlockPickingEnv(config):
  return ForceBlockPickingEnv(config)
