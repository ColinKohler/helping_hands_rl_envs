import numpy as np
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_picking_corner import CloseLoopBlockPickingCornerEnv

class ForceBlockPickingCornerEnv(CloseLoopBlockPickingCornerEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)
    finger_a_force, finger_b_force = self.robot.getFingerForce()

    #finger_force = [np.sqrt(np.sum(finger_a_force[:3])**2), np.sqrt(np.sum(finger_b_force[:3])**2)]
    finger_force = [finger_a_force[:3], finger_b_force[:3]]

    return state, hand_obs, obs, np.array(finger_force)

def createForceBlockPickingCornerEnv(config):
  return ForceBlockPickingCornerEnv(config)
