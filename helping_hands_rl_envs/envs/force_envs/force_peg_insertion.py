import numpy as np
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_peg_insertion import CloseLoopPegInsertionEnv

class ForcePegInsertionEnv(CloseLoopPegInsertionEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)
    finger_a_force, finger_b_force = self.robot.getFingerForce()

    finger_force = [finger_a_force[:3], finger_b_force[:3]]

    return state, hand_obs, obs, np.array(finger_force).reshape(-1)

def createForcePegInsertionEnv(config):
  return ForcePegInsertionEnv(config)
