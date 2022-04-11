import argparse

from helping_hands_rl_baselines.logger.multi_plotter import MultiPlotter

if __name__ == '__main__':
  parser =  argparse.ArgumentParser()
  args = parser.parse_args()

  log_filepaths = [
    [
      '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_force_1/log_data.pkl',
      '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_force_2/log_data.pkl',
      '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_force_3/log_data.pkl'
    ],
    [
      '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_1/log_data.pkl',
      '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_2/log_data.pkl',
      '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_3/log_data.pkl'
    ]
  ]
  log_names = ['vison+force', 'vision']

  base_dir = 'helping_hands_rl_baselines/scripts/outputs/'
  plotter = MultiPlotter(log_filepaths, log_names)
  plotter.plotLearningCurves('16x16 Block Picking', base_dir + 'train.pdf', 1500, window=100)
  plotter.plotEvalRewards('16x16 Block Picking', base_dir + 'eval_rewards.pdf', 35, window=2)
  #plotter.plotEvalReturns('128x128 Square Peg Insertion', base_dir +'eval_returns.pdf', window=5)
  #plotter.plotEvalLens('128x128 Square Peg Insertion', base_dir + 'eval_lens.pdf', window=5)
  #plotter.plotEvalValues('128x128 Square Peg Insertion', base_dir +'eval_values.pdf', window=5)
