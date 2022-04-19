import argparse

from helping_hands_rl_baselines.logger.multi_plotter import MultiPlotter

if __name__ == '__main__':
  parser =  argparse.ArgumentParser()
  args = parser.parse_args()

  log_filepaths = [
    [
      '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking_corner/128x128_runs/vision_force_4/log_data.pkl',
      '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking_corner/128x128_runs/vision_force_5/log_data.pkl',
      #'/home/colin/hdd/workspace/experiment_results/force_sac/block_picking_corner/128x128_runs/vision_force_6/log_data.pkl'
    ],
    [
      '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking_corner/128x128_runs/vision_1/log_data.pkl',
      '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking_corner/128x128_runs/vision_2/log_data.pkl',
      '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking_corner/128x128_runs/vision_3/log_data.pkl'
    ]
    #[
    #  '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_force_4/log_data.pkl',
    #  '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_force_5/log_data.pkl',
    #  '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_force_6/log_data.pkl'
    #],
    #[
    #  '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_1/log_data.pkl',
    #  '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_2/log_data.pkl',
    #  '/home/colin/hdd/workspace/experiment_results/force_sac/block_picking/16x16_runs/vision_3/log_data.pkl'
    #]
    #[
    #  '/home/colin/hdd/workspace/midichlorians/data/peg_insertion/md_tol_1/log_data.pkl',
    #  '/home/colin/hdd/workspace/midichlorians/data/peg_insertion/md_tol_2/log_data.pkl',
    #  '/home/colin/hdd/workspace/midichlorians/data/peg_insertion/md_tol_3/log_data.pkl',
    #],
    #[
    #  '/home/colin/hdd/workspace/ysalamir/data/peg_insertion/md_tol_4/log_data.pkl',
    #  '/home/colin/hdd/workspace/ysalamir/data/peg_insertion/md_tol_2/log_data.pkl',
    #  '/home/colin/hdd/workspace/ysalamir/data/peg_insertion/md_tol_3/log_data.pkl',
    #]
  ]
  log_names = ['vison+force', 'vision']

  base_dir = 'helping_hands_rl_baselines/scripts/outputs/'
  plotter = MultiPlotter(log_filepaths, log_names)
  plotter.plotLearningCurves('128x128 Lg Tol Peg Insertion', base_dir + 'train.pdf', 1500, window=100)
  plotter.plotEvalRewards('128x128 Lg Tol Peg Insertion', base_dir + 'eval_rewards.pdf', 40, window=5)
  plotter.plotEvalValues('128x128 Lg Tol Peg Insertion', base_dir +'eval_values.pdf', 40, window=5)
