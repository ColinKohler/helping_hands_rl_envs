import numpy as np
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_picking_corner import CloseLoopBlockPickingCornerEnv
from helping_hands_rl_envs.planners.close_loop_block_picking_corner_planner import CloseLoopBlockPickingCornerPlanner

class ForceBlockPickingCornerEnv(CloseLoopBlockPickingCornerEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)
    finger_a_force, finger_b_force = self.robot.getFingerForce()

    finger_force = [finger_a_force.tolist(), finger_b_force.tolist()]

    return state, hand_obs, obs, np.array(finger_force).reshape(-1)

def createForceBlockPickingCornerEnv(config):
  return ForceBlockPickingCornerEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.25, 0.65],
                          [-0.2, 0.2],
                          [0.01, 0.25]])

  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': None, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'panda',
                'object_init_space_check': 'point', 'physics_mode': 'force', 'object_scale_range': (1.2, 1.2),
                'hard_reset_freq': 1000, 'view_type': 'camera_center_xyz'}
  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi/8}
  env = ForceBlockPickingCornerEnv(env_config)
  planner = CloseLoopBlockPickingCornerPlanner(env, planner_config)

  num_success = 0
  for _ in range(20):
    obs = env.reset()
    done = False
    while not done:
      action = planner.getNextAction()

      print('Left Finger: x:{:.3f} y:{:.3f} z:{:.3f}'.format(obs[3][0], obs[3][1], obs[3][2]))
      print('Right Finger: x:{:.3f} y:{:.3f} z:{:.3f}'.format(obs[3][3], obs[3][4], obs[3][5]))
      print()
      plt.imshow(obs[2].squeeze(), cmap='gray'); plt.show()

      obs, reward, done = env.step(action)
    print(reward)
    if reward > 0.9:
      num_success += 1
  print(num_success)
