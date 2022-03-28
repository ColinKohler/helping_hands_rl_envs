import argparse

from helping_hands_rl_baselines.logger.plotter import Plotter

if __name__ == '__main__':
  parser =  argparse.ArgumentParser()
  args = parser.parse_args()

  log_filepaths = [
    '/home/colin/hdd/workspace/midichlorians/data/peg_insertion/clip_grad_norm/log_data.pkl',
    '/home/colin/hdd/workspace/midichlorians/data/peg_insertion/test/log_data.pkl',
    #'/home/colin/hdd/workspace/ysalamir/data/peg_insertion/32_x_32_obs/log_data.pkl',
  ]
  #log_names = ['force', 'vanilla']
  log_names = ['grad norm', 'no grad norm']

  plotter = Plotter(log_filepaths, log_names)
  plotter.plotLearningCurves('128x128 Square Peg Insertion', 'train.pdf', window=100, max_eps=None)
  plotter.plotEvalRewards('128x128 Square Peg Insertion', 'eval_rewards.pdf')
  plotter.plotEvalReturns('128x128 Square Peg Insertion', 'eval_returns.pdf')
  plotter.plotEvalLens('128x128 Square Peg Insertion', 'eval_lens.pdf')
