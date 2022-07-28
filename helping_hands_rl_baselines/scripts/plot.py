import argparse

from helping_hands_rl_baselines.logger.plotter import Plotter

if __name__ == '__main__':
  parser =  argparse.ArgumentParser()
  args = parser.parse_args()

  log_filepaths = [
    '/home/colin/hdd/workspace/midichlorians/data/drawer_opening/vision_force_1/log_data.pkl',
    '/home/colin/hdd/workspace/midichlorians/data/drawer_opening/new_weighted_loss/log_data.pkl',
    '/home/colin/hdd/workspace/ysalamir/data/drawer_opening/new/log_data.pkl',
  ]
  #log_names = ['vison+wrist_force', 'vision']
  log_names = ['old_force', 'new_force', 'new_vision']
  title = 'Drawer Opening'

  base_dir = 'helping_hands_rl_baselines/scripts/outputs/'
  plotter = Plotter(log_filepaths, log_names)
  plotter.plotLearningCurves(title, base_dir + 'train.pdf', window=100)
  plotter.plotEvalRewards(title, base_dir + 'eval_rewards.pdf', window=2, num_eval_intervals=20, eval_interval=500)
  plotter.plotEvalReturns(title, base_dir +'eval_returns.pdf', window=2, num_eval_intervals=20, eval_interval=500)
  plotter.plotEvalLens(title, base_dir + 'eval_lens.pdf', window=2)
  plotter.plotEvalValues(title, base_dir +'eval_values.pdf', window=2)
