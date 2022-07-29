import argparse

from helping_hands_rl_baselines.logger.plotter import Plotter

if __name__ == '__main__':
  parser =  argparse.ArgumentParser()
  args = parser.parse_args()

  log_filepaths = [
    #'/home/colin/hdd/workspace/midichlorians/data/block_picking_corner/new/log_data.pkl',
    '/home/colin/hdd/workspace/midichlorians/data/block_picking_corner/new_force_history_4/log_data.pkl',
    '/home/colin/hdd/workspace/ysalamir/data/block_picking_corner/new/log_data.pkl',
  ]
  #log_names = ['vison+wrist_force', 'vision']
  log_names = ['new_force', 'new_vision']
  title = 'Block Pulling'

  base_dir = 'helping_hands_rl_baselines/scripts/outputs/'
  plotter = Plotter(log_filepaths, log_names)
  plotter.plotLearningCurves(title, base_dir + 'train.pdf', window=100)
  plotter.plotEvalRewards(title, base_dir + 'eval_rewards.pdf', window=2, num_eval_intervals=20, eval_interval=500)
  plotter.plotEvalReturns(title, base_dir +'eval_returns.pdf', window=2, num_eval_intervals=20, eval_interval=500)
  plotter.plotEvalLens(title, base_dir + 'eval_lens.pdf', window=2)
  plotter.plotEvalValues(title, base_dir +'eval_values.pdf', window=2)
