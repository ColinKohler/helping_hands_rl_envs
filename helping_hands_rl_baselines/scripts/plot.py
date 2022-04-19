import argparse

from helping_hands_rl_baselines.logger.plotter import Plotter

if __name__ == '__main__':
  parser =  argparse.ArgumentParser()
  args = parser.parse_args()

  log_filepaths = [
    '/home/colin/hdd/workspace/midichlorians/data/peg_insertion/md_tol_1/log_data.pkl',
    '/home/colin/hdd/workspace/ysalamir/data/peg_insertion/md_tol_1/log_data.pkl',
  ]
  log_names = ['vison+force', 'vision']

  base_dir = 'helping_hands_rl_baselines/scripts/outputs/'
  plotter = Plotter(log_filepaths, log_names)
  plotter.plotLearningCurves('128x128 Square Peg Insertion', base_dir + 'train.pdf', window=100)
  plotter.plotEvalRewards('128x128 Square Peg Insertion', base_dir + 'eval_rewards.pdf', window=2)
  plotter.plotEvalReturns('128x128 Square Peg Insertion', base_dir +'eval_returns.pdf', window=2)
  plotter.plotEvalLens('128x128 Square Peg Insertion', base_dir + 'eval_lens.pdf', window=2)
  plotter.plotEvalValues('128x128 Square Peg Insertion', base_dir +'eval_values.pdf', window=2)
